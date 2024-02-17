# Contributing to the biondeep-IG documentation website

When **creating/updating the documentation pages**, you should first follow the bellow steps to create the documentation Docker container.
It will allow you to visualize your modifications to the documentation website locally on your navigator and in real time before pushing any changes.
The local address is [`127.0.0.1:8000`](`http://127.0.0.1:8000`)

>- 🚨 You must not rely on the IDE code preview to visualize the documentation Markdown (.MD) pages. Code editors interpret the MD code in a different way from `mkdocs` that we use to create this documentation website.
>- Relying on IDE preview results in inconsistencies between the edited documentation pages locally and the published pages on the documentation website. (Table of content are changed, code blocks are mixed with previous text, ordered lists don't show correct numbering, …)

## Installing the documentation Docker container

1. First, make sure you have [docker installed](installation.md#docker-installation).

2. Next, open a new terminal inside the `biondeep-ig` folder

      ```bash
      cd path/to/biondeep-ig
      ```

3. Then, you need to **build**, **create** and **start** the documentation Docker container:

      ```bash
      make docs
      ```

4. Finally, you can visualize the documentation in your local navigator using the default address: [`127.0.0.1:8000`](`http://127.0.0.1:8000`)

## Notes

- The documentation website is updated automatically when you change any Markdown (.md) file.
- The mkdocs server uses by default the `8000` port, you can modify it using the variable `MKDOCS_PORT`
      ```bash
      make docs MKDOCS_PORT=80
      ```
- If the container or the mkdocs server crashes or the terminal is closed , the container will be **automatically deleted** and you have to [**recreate it**](#installing-the-documentation-docker-container).

## How it works

- The command `make docs` **builds**, **creates** and **starts** the documentation Docker container in the current terminal following the `Dockerfile.docs`
- Inside the docker, the `mkdocs` server is automatically launched using `mkdocs serve -a 0.0.0.0:8000`. (It is necessary to add the address `0.0.0.0` so the docker port forwarding works correctly)
- Then the user can visualize the documentation website by simply opening the default adresse in a local navigator [`127.0.0.1:8000`](`http://127.0.0.1:8000`)
