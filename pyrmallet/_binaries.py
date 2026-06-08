import shutil


def find_binary(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(
            f"'{name}' not found on PATH. "
            "Install RustMallet binaries via 'cargo build --release' "
            "and add target/release/ to PATH, or download from "
            "https://github.com/mimno/RustMallet/releases"
        )
    return path
