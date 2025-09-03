# Dev Loop Reminders

## Git Commit Squashing

### View last k commits
```bash
git log --oneline -k
```

### Squash last k commits
```bash
git reset --soft HEAD~k && git commit
```

Replace `k` with the number of commits you want to work with.

## Quick Git Add and Commit

### Add all Python files and commit
```bash
find . -type f -name "*.py" | xargs git add && git commit -m "update"
```