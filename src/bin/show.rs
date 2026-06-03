/// Display trained topic model results in human-readable form.
/// Reads the TSV files written by `train` and prints formatted output.
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::{self, BufRead};
use std::path::Path;

fn print_usage() {
    eprintln!(
        "Usage: show [options]

Options:
  --topic-word <file>   Topic-word probability file (default: topic_word.tsv)
  --doc-topic <file>    Document-topic probability file (default: doc_topic.tsv)
  --corpus <file>       Binary corpus file for document text excerpts
  --words <n>           Words to show per topic (default: 10)
  --top-docs <n>        Top documents to show per topic (default: 5, 0 = off)
  --top-labels <n>      Top labels to show per topic (default: 5, 0 = off)
  --doc-topics <n>      [Legacy] Top topics per document (default: 0 = off)
  --threshold <f>       [Legacy] Min probability for --doc-topics (default: 0.1)
"
    );
}

struct Args {
    topic_word: String,
    doc_topic:  String,
    corpus:     Option<String>,
    words:      usize,
    top_docs:   usize,
    top_labels: usize,
    doc_topics: usize,
    threshold:  f64,
}

impl Default for Args {
    fn default() -> Self {
        Args {
            topic_word: "topic_word.tsv".to_string(),
            doc_topic:  "doc_topic.tsv".to_string(),
            corpus:     None,
            words:      10,
            top_docs:   5,
            top_labels: 5,
            doc_topics: 0,
            threshold:  0.1,
        }
    }
}

fn parse_args() -> Option<Args> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut args = Args::default();
    let mut i = 0;
    while i < raw.len() {
        match raw[i].as_str() {
            "--topic-word"  => { i += 1; args.topic_word  = raw[i].clone(); }
            "--doc-topic"   => { i += 1; args.doc_topic   = raw[i].clone(); }
            "--corpus"      => { i += 1; args.corpus      = Some(raw[i].clone()); }
            "--words"       => { i += 1; args.words       = raw[i].parse().ok()?; }
            "--top-docs"    => { i += 1; args.top_docs    = raw[i].parse().ok()?; }
            "--top-labels"  => { i += 1; args.top_labels  = raw[i].parse().ok()?; }
            "--doc-topics"  => { i += 1; args.doc_topics  = raw[i].parse().ok()?; }
            "--threshold"   => { i += 1; args.threshold   = raw[i].parse().ok()?; }
            "--help" | "-h" => return None,
            other => { eprintln!("Unknown argument: {}", other); return None; }
        }
        i += 1;
    }
    Some(args)
}

// ---------------------------------------------------------------------------
// Topic-word loading with exclusivity
// ---------------------------------------------------------------------------

struct WordEntry {
    word: String,
    /// φ(w,t)² / Σₖ φ(w,k) — high when word is both probable and exclusive to this topic.
    excl: f64,
}

fn load_topic_word(path: &Path) -> io::Result<BTreeMap<usize, Vec<WordEntry>>> {
    let file = fs::File::open(path)?;
    let reader = io::BufReader::new(file);

    let mut raw: BTreeMap<usize, Vec<(String, f64)>> = BTreeMap::new();
    let mut word_total: HashMap<String, f64> = HashMap::new();

    for (line_idx, line) in reader.lines().enumerate() {
        let line = line?;
        if line_idx == 0 { continue; }
        let cols: Vec<&str> = line.splitn(3, '\t').collect();
        if cols.len() < 3 { continue; }
        let topic: usize = match cols[0].parse() {
            Ok(t) => t,
            Err(_) => continue,
        };
        let word = cols[1].to_string();
        let prob: f64 = cols[2].parse().unwrap_or(0.0);
        *word_total.entry(word.clone()).or_insert(0.0) += prob;
        raw.entry(topic).or_default().push((word, prob));
    }

    let mut result: BTreeMap<usize, Vec<WordEntry>> = BTreeMap::new();
    for (topic, mut words) in raw {
        words.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let entries = words.into_iter().map(|(word, prob)| {
            let total = word_total.get(&word).copied().unwrap_or(prob);
            WordEntry { word, excl: prob * prob / total }
        }).collect();
        result.insert(topic, entries);
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Document-topic matrix
// ---------------------------------------------------------------------------

struct DocTopicMatrix {
    doc_names:  Vec<String>,
    doc_labels: Vec<String>,
    has_labels: bool,
    /// probs[doc][topic]
    probs:      Vec<Vec<f64>>,
    num_topics: usize,
}

fn load_doc_topic_matrix(path: &Path) -> io::Result<DocTopicMatrix> {
    let file = fs::File::open(path)?;
    let reader = io::BufReader::new(file);
    let mut lines = reader.lines();

    let header = match lines.next() {
        Some(Ok(h)) => h,
        _ => return Ok(DocTopicMatrix {
            doc_names: vec![], doc_labels: vec![], has_labels: false, probs: vec![], num_topics: 0,
        }),
    };

    let cols: Vec<&str> = header.split('\t').collect();
    let first_topic_col = if cols.get(1).map(|&c| c == "label").unwrap_or(false) { 2 } else { 1 };
    let has_labels = first_topic_col == 2;
    let num_topics = cols.len().saturating_sub(first_topic_col);

    let mut doc_names  = Vec::new();
    let mut doc_labels = Vec::new();
    let mut probs      = Vec::new();

    for line in lines {
        let line = line?;
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < first_topic_col + num_topics { continue; }

        doc_names.push(fields[0].to_string());
        doc_labels.push(if has_labels { fields[1].to_string() } else { String::new() });

        let row: Vec<f64> = fields[first_topic_col..first_topic_col + num_topics]
            .iter()
            .map(|&v| v.parse().unwrap_or(0.0))
            .collect();
        probs.push(row);
    }

    Ok(DocTopicMatrix { doc_names, doc_labels, has_labels, probs, num_topics })
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/// Effective document count: 1 / HHI of normalised per-doc topic probabilities.
/// Low = topic concentrated in a few docs (focused); high = spread across many (background).
fn effective_doc_count(topic_probs: &[f64]) -> f64 {
    let sum: f64 = topic_probs.iter().sum();
    if sum == 0.0 { return 0.0; }
    let hhi: f64 = topic_probs.iter().map(|&p| (p / sum).powi(2)).sum();
    1.0 / hhi
}

/// Mean θ(d,t) per label. Using mean rather than sum so large label groups
/// don't crowd out small but topically coherent ones.
fn top_labels_for_topic(
    doc_labels:  &[String],
    topic_probs: &[f64],
    n:           usize,
) -> Vec<(String, f64)> {
    let mut label_sum:   HashMap<&str, f64>   = HashMap::new();
    let mut label_count: HashMap<&str, usize> = HashMap::new();
    for (label, &prob) in doc_labels.iter().zip(topic_probs.iter()) {
        if label.is_empty() { continue; }
        *label_sum.entry(label.as_str()).or_insert(0.0) += prob;
        *label_count.entry(label.as_str()).or_insert(0) += 1;
    }
    let mut scores: Vec<(String, f64)> = label_sum.iter()
        .map(|(&l, &s)| {
            let count = label_count.get(l).copied().unwrap_or(1) as f64;
            (l.to_string(), s / count)
        })
        .collect();
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scores.truncate(n);
    scores
}

// ---------------------------------------------------------------------------
// Corpus text reconstruction
// ---------------------------------------------------------------------------

fn load_corpus_texts(corpus_path: &Path) -> io::Result<HashMap<String, String>> {
    let corpus = rust_mallet::corpus::load_corpus(corpus_path)?;
    let texts = corpus.doc_names.iter().zip(corpus.docs.iter())
        .map(|(name, doc)| {
            let text = doc.iter()
                .map(|&id| corpus.id_to_word[id as usize].as_str())
                .collect::<Vec<_>>()
                .join(" ");
            (name.clone(), text)
        })
        .collect();
    Ok(texts)
}

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

fn show_topics(
    topics:       &BTreeMap<usize, Vec<WordEntry>>,
    doc_matrix:   Option<&DocTopicMatrix>,
    corpus_texts: Option<&HashMap<String, String>>,
    n_words:      usize,
    n_top_docs:   usize,
    n_top_labels: usize,
) {
    let num_topics = topics.len();
    let width = num_topics.to_string().len();
    let num_docs = doc_matrix.map(|dm| dm.probs.len()).unwrap_or(0);

    for (&topic_idx, words) in topics {
        let topic_probs: Option<Vec<f64>> = doc_matrix
            .map(|dm| dm.probs.iter().map(|row| row[topic_idx]).collect());

        let eff = topic_probs.as_deref().map(effective_doc_count);

        // Header line
        print!("Topic {:>width$}", topic_idx, width = width);
        if let Some(e) = eff {
            print!("  [{:.0} of {} effective documents]", e, num_docs);
        }
        println!(":");

        // Top words by probability
        let top_prob: Vec<&str> = words.iter().take(n_words).map(|w| w.word.as_str()).collect();
        println!("  Prob: {}", top_prob.join("  "));

        // Top words by probability-weighted exclusivity
        let mut by_excl: Vec<&WordEntry> = words.iter().collect();
        by_excl.sort_by(|a, b| b.excl.partial_cmp(&a.excl).unwrap_or(std::cmp::Ordering::Equal));
        let top_excl: Vec<&str> = by_excl.iter().take(n_words).map(|w| w.word.as_str()).collect();
        println!("  Excl: {}", top_excl.join("  "));

        if let (Some(dm), Some(tp)) = (doc_matrix, topic_probs.as_deref()) {
            // Top labels (mean θ per label)
            if n_top_labels > 0 && dm.has_labels {
                let labels = top_labels_for_topic(&dm.doc_labels, tp, n_top_labels);
                if !labels.is_empty() {
                    let label_strs: Vec<String> = labels.iter()
                        .map(|(l, s)| format!("{} ({:.0}%)", l, s * 100.0))
                        .collect();
                    println!("  Labels: {}", label_strs.join("  "));
                }
            }

            // Top documents
            if n_top_docs > 0 {
                let mut doc_order: Vec<(usize, f64)> = tp.iter().copied().enumerate().collect();
                doc_order.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                println!("  Top documents:");
                for (doc_idx, prob) in doc_order.iter().take(n_top_docs) {
                    let name = &dm.doc_names[*doc_idx];
                    let excerpt: String = corpus_texts
                        .and_then(|ct| ct.get(name))
                        .map(|t| t.chars().take(150).collect())
                        .unwrap_or_default();
                    if excerpt.is_empty() {
                        println!("    ({:.0}%)  {}", prob * 100.0, name);
                    } else {
                        println!("    ({:.0}%)  {}:  {}", prob * 100.0, name, excerpt);
                    }
                }
            }
        }

        println!();
    }
}

fn show_doc_topics(path: &Path, n_topics: usize, threshold: f64) -> io::Result<()> {
    let file = fs::File::open(path)?;
    let reader = io::BufReader::new(file);
    let mut lines = reader.lines();

    let header = match lines.next() {
        Some(Ok(h)) => h,
        _ => return Ok(()),
    };
    let cols: Vec<&str> = header.split('\t').collect();

    let first_topic_col = if cols.get(1).map(|&c| c == "label").unwrap_or(false) { 2 } else { 1 };
    let has_label = first_topic_col == 2;
    let n_topic_cols = cols.len() - first_topic_col;

    println!();
    println!("Document topics (threshold {:.0}%):", threshold * 100.0);
    println!();

    for line in lines {
        let line = line?;
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < first_topic_col + n_topic_cols { continue; }

        let doc_name = fields[0];
        let label    = if has_label { Some(fields[1]) } else { None };

        let mut probs: Vec<(usize, f64)> = fields[first_topic_col..]
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| {
                let p: f64 = v.parse().ok()?;
                Some((i, p))
            })
            .collect();

        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top: Vec<String> = probs
            .iter()
            .take(n_topics)
            .filter(|(_, p)| *p >= threshold)
            .map(|(t, p)| format!("{} ({:.0}%)", t, p * 100.0))
            .collect();

        if top.is_empty() { continue; }

        match label {
            Some(lbl) => println!("{} [{}]:  {}", doc_name, lbl, top.join(", ")),
            None       => println!("{}:  {}", doc_name, top.join(", ")),
        }
    }

    Ok(())
}

fn main() {
    let args = match parse_args() {
        Some(a) => a,
        None => { print_usage(); std::process::exit(1); }
    };

    let tw_path = Path::new(&args.topic_word);
    if !tw_path.exists() {
        eprintln!(
            "Error: '{}' not found. Run `train` first, or specify a file with --topic-word.",
            args.topic_word
        );
        std::process::exit(1);
    }

    let topics = match load_topic_word(tw_path) {
        Ok(t) => t,
        Err(e) => { eprintln!("Error reading {}: {}", args.topic_word, e); std::process::exit(1); }
    };

    let dt_path = Path::new(&args.doc_topic);
    let doc_matrix = if dt_path.exists() {
        match load_doc_topic_matrix(dt_path) {
            Ok(m) if m.num_topics > 0 => Some(m),
            Ok(_) => None,
            Err(e) => {
                eprintln!("Warning: error reading {}: {}", args.doc_topic, e);
                None
            }
        }
    } else {
        None
    };

    let corpus_texts = match args.corpus.as_deref() {
        Some(cp) => match load_corpus_texts(Path::new(cp)) {
            Ok(t)  => Some(t),
            Err(e) => { eprintln!("Warning: error reading corpus {}: {}", cp, e); None }
        },
        None => None,
    };

    show_topics(
        &topics,
        doc_matrix.as_ref(),
        corpus_texts.as_ref(),
        args.words,
        args.top_docs,
        args.top_labels,
    );

    if args.doc_topics > 0 && dt_path.exists() {
        if let Err(e) = show_doc_topics(dt_path, args.doc_topics, args.threshold) {
            eprintln!("Error reading {}: {}", args.doc_topic, e);
        }
    }
}
