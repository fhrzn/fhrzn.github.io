# Personal website by Hugo and PaperMod theme 🚀

## How to Edit
1. Clone this repository `git clone https://github.com/fhrzn/personal-site.git`.
2. Update theme submodule `git submodule update --remote --merge`.
3. Run `hugo new --kind posts <subfolder>/<page-name.md>` to create new post page.
4. Run `hugo` to build website.
5. Deploy! 🚀


## How to Create New Page
Run command:
```bash
hugo new content <path>/<filename>.md
```
OR
```bash
hugo new --kind posts <subfolder>/<page-name.md>
```
Example:
```bash
hugo new content posts/my-new-page/index.md
```

## How to Run
Run command:
```bash
hugo server
```
And it will start in the following path:
```bash
http://localhost:1313/
``` 