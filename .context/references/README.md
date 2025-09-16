# References Directory

- This directory is primarily used for reference by Cursor, Claude Code and Windsurf. To avoid disrupting their indexing context and prevent damage to the code project structure, it needs to be added to .gitignore.
- We provide an automated script `scripts/update-references.sh` to manage reference projects.

## Usage

1. Add GitHub repository links to clone in the `.references/references-list.txt` file, one per line. For example:

   ```txt
   https://github.com/vercel/ai.git
   https://github.com/vercel/ai-chatbot.git
   ```

2. Run the automated script:

   ```shell
   ./scripts/update-references.sh
   ```

3. The script will automatically perform the following operations:
   - Clone or update repositories in the `.references` directory for each link
   - Automatically add repository paths to the `.gitignore` file
   - Ensure each entry occupies a separate line in `.gitignore`

## Manual Management (Not Recommended)

If you need to manually manage reference projects, follow these steps:

```shell
# Clone project to .references directory
git clone https://github.com/vercel/ai.git .references/ai

# Then ensure the corresponding entry is added to .gitignore
echo ".references/ai" >> .gitignore
```
