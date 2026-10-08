# Data policy & removal (AERIS / NeuroField)

**Contact for all data issues:** giriisdev@gmail.com

## What data is in this repo?

| Location | Nature |
|----------|--------|
| `data/*.txt`, `data/align/*` | Short **demo / synthetic / author-written** text for tiny training experiments |
| `docs/*/model.pt` etc. | **Experimental weights** trained on that demo text (not production models) |
| No intentional collection of end-user PII in the training samples shipped here |

This project does **not** claim to ship Common Crawl dumps, copyrighted books in full, or private user databases.

## Your responsibility when you train

If **you** add datasets (books, dumps, customer chats, scraped sites):

1. Ensure you have the **right** to use that data (license, permission, or public domain).
2. Do not commit secrets, passwords, API keys, or other people’s private data.
3. Prefer public, licensed corpora and document the source in your own notes.

The author is **not responsible** for datasets you add locally or push from your fork.

## Data removal request

If you believe a file in **this** repository contains material you own or personal data that should not be public:

1. Email **giriisdev@gmail.com** with:
   - Subject: `AERIS data removal request`
   - File path(s) (e.g. `data/example.txt`)
   - Reason (copyright / personal data / other)
   - Proof of authority if applicable
2. Author will review and, if appropriate:
   - Delete or replace the file in the repository
   - Commit message will note the removal
3. Git history may still contain old blobs until history is rewritten; for sensitive cases ask for history cleanup in the same email.

## Local “right to remove” on your machine

```bash
# remove a data file you no longer want
rm data/some_file.txt

# optional: retrain without it, then delete old checkpoint
rm -rf docs/AERIS_64
```

Never put production user logs into `data/` without a privacy process.

## Models

Removing training text does **not** automatically erase what a model already memorised in an old checkpoint. Delete the checkpoint too if that matters:

```bash
rm -rf docs/AERIS_64 docs/neurofield_*
```

## Disclaimer

This policy supports good practice for a personal open repo. It is **not** a GDPR/CCPA legal compliance package for a company product. If you ship a commercial service, get proper legal counsel.
