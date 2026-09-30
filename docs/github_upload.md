# GitHub Upload Guide

## Create the Repository

Create an empty public repository named `Terra-JDR` under the GitHub account
`lvhui-7150`. Do not initialize it with a README, license, or `.gitignore`.

## Initialize and Commit

Run from this directory:

```powershell
git init
git add .
git commit -m "Initial release of Terra-JDR code and data"
git branch -M main
git remote add origin https://github.com/lvhui-7150/Terra-JDR.git
git push -u origin main
```

## Before Publishing

1. Add a `LICENSE` file if reuse permissions are intended.
2. Verify that no manuscript or PDF files are present.
3. Run:

```powershell
rg --files | rg "\.(pdf|docx|tex)$"
```

The command should return no manuscript-related files.
