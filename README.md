# Guida all'utilizzo di JUPYTEXT
### Installazione 
È semplicissimo, basta eseguire una di queste due linee di codice: <br>

- se si usa il _package manager "standard"_ di python:
  ```console
  pip install jupytext
  ```
- se si usa il _package manager_ Anaconda:
  ```console
  conda install jupytext -c conda-forge
  ```

### Utilizzo
__Perché usiamo jupytext?__ Per poter fare un'operazione di versioning più ragionevole:<br>
se facessimo il versioning utilizzando i notebook _.ipynb_ le differenze da una versione all'altra risulterebbero meno visibili, <br>
con ___jupytext___ si vanno a colmare queste mancanze dei notebook.

__Conversione:__ <br>

- da _notebook_ a file _.py_:
  ```console
  jupytext --to py notebook.ipynb 
  ```
- da file _.py_ a _notebook_:
  ```console
  jupytext --to notebook notebook.py
  ```

Reference: [JupytextCLI](https://jupytext.readthedocs.io/en/latest/using-cli.html)
