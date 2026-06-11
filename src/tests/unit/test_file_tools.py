import pytest

from src.shared.exceptions import ToolError
from src.tools import file_tools


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "train.py").write_text("print('train')")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/main")
    (tmp_path / "requirements.txt").write_text("scikit-learn\n")
    return str(tmp_path)


def test_scan_excludes_git_internals(repo):
    files = file_tools.scan_repo(repo)
    assert "src/train.py" in files
    assert "requirements.txt" in files
    assert not any(".git" in f for f in files)


def test_read_and_write_roundtrip(repo):
    file_tools.write_file(repo, "aml/environment.yml", "name: env")
    assert file_tools.read_file(repo, "aml/environment.yml") == "name: env"


def test_path_escape_rejected(repo):
    with pytest.raises(ToolError):
        file_tools.write_file(repo, "../outside.txt", "nope")
    with pytest.raises(ToolError):
        file_tools.read_file(repo, "../../etc/passwd")


def test_read_key_files(repo):
    contents = file_tools.read_key_files(repo)
    assert "requirements.txt" in contents
    assert "scikit-learn" in contents["requirements.txt"]
