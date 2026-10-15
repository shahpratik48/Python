# gitlab_dump (Tool 2)

Pulls all code from the `main` branch of the ikg / odm / nlg GitLab projects into
txt files, masks them, restores masked files, and rebuilds the real files from the txt.

## Files
- `config.py` – GitLab URL, the three project paths, branch (`main`), output folder, `MASK_SHIFT`, suffixes
- `gitlab_dump.py` – the code; run from command line or import `run()`
- `gitlab_dump.ipynb` – same as a notebook (config + code + run cell)

## Features (run any, skip any)
| step   | input                      | output                                    |
|--------|----------------------------|-------------------------------------------|
| export | GitLab (token asked)       | i.txt, o.txt, n.txt                       |
| mask   | i.txt, o.txt, n.txt        | i_masked.txt, o_masked.txt, n_masked.txt  |
| unmask | *_masked.txt (same folder) | i_restored.txt, o_restored.txt, n_restored.txt (or i/o/n.txt with restore_to_original=True) |
| rebuild | *_restored.txt (or i/o/n.txt with rebuild_source="original") | real files under rebuilt/ikg/, rebuilt/odm/, rebuilt/nlg/ in the original folder structure, e.g. rebuilt/ikg/dags/ikg/test/test.sql, plus ikg.zip / odm.zip / nlg.zip in the output folder |
| verify | i/o/n.txt, *_restored.txt, rebuilt/ | report: byte-by-byte comparison of original vs restored txt, and of every rebuilt file vs the original txt; first difference shown with line/column |

    pip install requests
    python gitlab_dump.py                 # asks y/n for each step
    python gitlab_dump.py export mask     # run only these
    python gitlab_dump.py unmask
    python gitlab_dump.py unmask rebuild verify
    python gitlab_dump.py verify

## txt format
    ****test.sql******
    ****ikg-dags/dags/ikg/test/test.sql******

    <complete file content, unchanged>

    -----------------------------------------------------------------------------------------------------------------------------

Every file in every folder/subfolder is included (recursive tree listing with pagination).
Binary files get a one-line placeholder instead of content.

## Content fidelity
File content is copied exactly as it is in GitLab at every stage (txt, masked, restored,
rebuilt): line endings (LF/CRLF), tabs, indentation, trailing spaces, a missing final
newline, upper/lower case, digits, special and non-ASCII characters are all preserved.
All files are read and written with `newline=""` and UTF-8 so nothing is translated.
Masking changes only ASCII letters and nothing else. The rebuilt files are byte-identical
to the originals (verified with tabs, CRLF, unicode, no-trailing-newline, empty files, and
files containing separator-like lines).

## Masking
Every ASCII letter is shifted by `MASK_SHIFT` (config.py, default 1), case kept (shift 1: a->b, z->a, A->B, Z->A);
digits, punctuation, whitespace and non-ASCII characters are unchanged. Unmask is the reverse shift.

## Rebuild
Each block in the txt is parsed back into (name, path, code). The leading project folder
in the path (`ikg-dags/`) is replaced by the project key folder (`ikg/`), all intermediate
folders are created, and the file is written with its content unchanged. Blocks that were
binary placeholders are skipped and counted.

## Notes
- Token needs `read_api` (or `api`) scope with at least Reporter access on the three projects.
- Behind an internal CA set `VERIFY_SSL` in `config.py` to the CA bundle path.
