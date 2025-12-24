# 📤 How to Upload this Folder to GitHub

Follow these steps to turn this folder into a GitHub repository.

## 1. Prerequisites
- Ensure you have **Git** installed on your computer.
- Ensure you have a **GitHub Account**.

## 2. Create the Repository on GitHub
1. Go to [github.com/new](https://github.com/new).
2. **Repository name**: `mlProjects_Mehdi` (or any name you prefer).
3. **Description**: "Portfolio of 20 Data Science Projects".
4. **Public/Private**: Choose Public.
5. **Initialize this repository with**: DO NOT check any boxes (No README, No .gitignore). We will import our own.
6. Click **Create repository**.

## 3. Initialize Git Locally (Terminal)
Open your terminal (PowerShell or Command Prompt) and navigate to this folder:

```powershell
cd "c:\Users\Mehdi\OneDrive\Desktop\prog\DataScience_Google\mlProjects_Mehdi"
```

Run the following commands one by one:

### A. Initialize
```bash
git init
```

### B. Add Files
```bash
git add .
```
*(This stages all 20 projects and this README for commit)*

### C. Commit
```bash
git commit -m "Initial commit: Added 20 Data Science projects"
```

## 4. Connect to GitHub and Push
Copy the HTTPS URL from the GitHub page you just created (e.g., `https://github.com/YourUsername/mlProjects_Mehdi.git`).

Run these commands (replace the URL with yours):

```bash
# 1. Rename branch to main (best practice)
git branch -M main

# 2. Add the remote link
git remote add origin https://github.com/YourUsername/mlProjects_Mehdi.git

# 3. Push your code
git push -u origin main
```

## 5. Success!
Refresh your GitHub page. You should see all your folders (`Projet_01`...) and the nicely formatted `README.md`.
