# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Machine Learning Vademecum
# ## Introduzione
# Il machine learning (ML) è la disciplina il cui obiettivo è **migliorare le prestazioni di un sistema apprendendo dall'esperienza tramite metodi computazionali**.  
#
# Non è poi così diverso da quello che anche noi umani facciamo quotidianamente: di continuo ci basiamo su esperienze pregresse per fare predizioni più o meno accurate sulla realtà. 
# Ad esempio, sappiamo prevedere con una certa sicurezza se una pesca sia buona o meno, semplicemente guardandola e toccandola, ancor prima di mangiarla.  
# La predizione non deriva da un ragionamento logico comprovato (altrimenti sarebbe sempre esatta), ma dall'essersi scontrati già più volte con le istanze del problema. 
# Questo ci ha fornito, tramite quello che è un **processo induttivo**, un meccanismo risolutivo col quale affrontare istanze nuove, mai incontrate prima.
#
# Seguendo questo paradigma, l'idea alla base del ML è dare in pasto dei **dati** ad un cosiddetto **algoritomo di apprendimento**, che li usi per produrre un **modello** in grado di fornire risposte sufficientemente accurate di fronte a nuove istanze del problema.  
# Esattamente come le nostre predizioni possono rivelarsi errate, così possono esserlo le risposte fornite dal modello, al quale si associano quindi delle **metriche** per valutarne le performance.  
#
# Ma perché accontentarsi di una risposta *probabilmente* corretta al posto di una la cui correttezza è dimostrabile logicamente?
#
# ### Due scenari
# #### 1. Il problema della nonna
# Esistono problemi per i quali **non siamo in grado di formulare una soluzione**, anche perché può non esistere.  
# Per una persona, stabilire se una foto ritragga o meno sua nonna è un compito triviale; per un computer non lo è affatto. Questo è un esempio di problema "_non risolvibile_", per cui ricorriamo al ML.
#
# #### 2. Il problema dello zaino
# Per alcuni problemi, formulare una soluzione è del tutto fattibile (se non facile), ma questa ha un **costo computazionale proibitivo**.  
# Nel <a href="https://it.wikipedia.org/wiki/Problema_dello_zaino">problema dello zaino</a>, la soluzione _brute force_ è concettualmente semplice: basta considerare tutte le possibili combinazioni di oggetti e scegliere quella con il valore massimo senza superare il peso limite. Tuttavia, questa strategia diventa impraticabile al crescere del numero di oggetti ($n$): ad esempio, con $n=50$, si arriva a oltre un milione di miliardi di combinazioni da esaminare ($2^{50}$) .
#
# In queste due situazioni ricorriamo ad un modello di ML, nell'ottica che le sue risposte approssimino bene quelle esatte, se il problema ne prevede.

# ## Modalità di apprendimento
# In base a come si presentano i dati che l’algoritmo di apprendimento usa per produrre il modello, distinguiamo tra apprendimento **supervisionato** e non.
#
# ### Apprendimento supervisionato
# Ogni osservazione è rappresentata da una coppia $(x, y)$, dove $x$ è l'istanza del problema e $y$ è la relativa soluzione (o etichetta).  
# Ad esempio, se $x$ è una foto della nonna da classificare, $y$ sarà la risposta _sì/no_.  
# Il dataset è quindi un insieme di coppie _istanza-soluzione_: $\{(x_1, y_1), (x_2, y_2), \dots, (x_n, y_n)\}$.  
# Chiaramente deve esistere una relazione $f$ che leghi istanze e soluzioni, tale che $f(x) = y$. L'obiettivo del modello è proprio approssimare $f$, che prende il nome di _ground truth_. Le predizioni del modello vengono indicate come $\hat{y}$.
#
# ### Apprendimento non supervisionato
# Ogni osservazione è costituita esclusivamente dall'istanza del problema: $(x)$.  
# Un esempio di apprendimento non supervisionato è il _clustering_, in cui il modello deve raggruppare i dati in insiemi (_cluster_) basati su somiglianze o caratteristiche comuni. L'obiettivo è che le osservazioni all'interno di ciascun gruppo siano più simili tra loro rispetto a quelle di altri gruppi, permettendo di individuare pattern e strutture nascoste nei dati senza l'utilizzo delle etichette.
#
# **_Nota_**: il notebook si concentra sul ML supervisionato.

# + [markdown] jp-MarkdownHeadingCollapsed=true
# ## Valutare un modello
# Poiché l'obiettivo è approssimare la relazione di base (_ground truth_) tra istanze e soluzioni, è necessario **valutare la bontà di tale approssimazione**, ossia valutare il modello.  
# L'idea di base è molto semplice: si forniscono al modello delle osservazioni (coppie $(x, y)$), e per ciascuna si confronta la risposta prodotta dal modello, $\hat{y}$, con la soluzione reale, $y$.  
# A partire da questo confronto, è possibile monitorare diverse **metriche** di valutazione delle prestazioni.
#
# ### Alcune metriche
# Presentiamo alcune delle metriche che vengono comunemente impiegate per la valutazione delle prestazioni.  
# Notazione:
# - $D$ : dataset, composto da esempi del tipo $(x_i, y_i)$
# - $m = |D|$
# - $f$ : modello; $f(x_i)$ è l'output del modello quando sottoposto all'input (istanza) $x_i$
# - $\mathbb{I}$ : <a href="https://it.wikipedia.org/wiki/Funzione_indicatrice">funzione indicatrice</a>
#
# #### MSE
# L'errore quadratico medio (Mean Squared Error) è tipicamente utilizzato nei problemi di regressione (come quello descritto nel paragrafo successivo).
# $$
# MSE(f;D)= \frac{1}{m}\sum_{i=1}^{m} (f(x_i) - y_i)^2
# $$
#
# #### Tasso d'errore (_error rate_) e accuratezza (_accuracy_)
# Sono le due metriche più diffuse nei problemi di classificazione.  
# Il tasso d'errore indica la proporzione di istanze erroneamente classificate sul totale delle istanze.
# $$
# \operatorname{E}(f;D)=\frac{1}{m}\sum_{i=1}^{m}\mathbb{I}(f(x_i)\neq y_i)
# $$
#
# L'accuratezza indica la proporzione di istanze correttamente classificate sul totale delle istanze.
# $$
# \operatorname{acc}(f;D)=\frac{1}{m}\sum_{i=1}^{m}\mathbb{I}(f(x_i)=y_i)
# $$
# Valgono anche:
# $$
# \operatorname{acc}(f;D)=1-\operatorname{E}(f;D)
# $$
# $$
# \operatorname{E}(f;D)=1-\operatorname{acc}(f;D)
# $$
#
# #### Precisione (_precision_) e richiamo (_recall_)
# Nel sottocaso dei problemi di classificazione binaria, si tratta di due metriche molto comuni e più informative rispetto a tasso d'errore o accuratezza.  
# La precisione indica, in rapporto, quanti dei campioni previsti come positivi sono effettivamente positivi.
# $$
# \frac{\text{Veri positivi (TP)}}{\text{Veri positivi (TP)} + \text{Falsi positivi (FP)}}
# $$
# La si può vedere come una misura dell'accuratezza delle previsioni positive.
#
# Il richiamo indica la proporzione di esempi positivi reali che sono stati correttamente identificati dal modello.
# $$
# \frac{\text{Veri positivi (TP)}}{\text{Veri positivi (TP)} + \text{Falsi negativi (FN)}}
# $$
# Lo si può vedere come una misura della capacità del modello di trovare tutti i positivi.  
#
# Importante notare come precisione e richiamo siano in mutua "competizione". In generale, il richiamo è spesso basso quando la precisione è alta, e la precisione è spesso bassa quando il richiamo è alto.  
# Infatti, se volessimo facilmente portare a 1 il richiamo del nostro modello, basterebbe far sì che questo classifichi positivamente ogni istanza, portando logicamente a 0 i falsi negativi. Ma questo andrebbe a scapito della precisione, per via dei numerosi ed inevitabili falsi positivi. Al contrario, classificando come positive solo le istanze di cui si è quasi certi, dunque massimizzando la precisione, si genererebbero parecchi falsi negativi che inciderebbero negativamente sul richiamo.  
# Tipicamente, possiamo ottenere alti livelli per entrambe le metriche solo nei problemi più semplici. In quelli complessi, dobbiamo capire quale metrica è più importante.
# Ecco due esempi significativi:
# - In un sistema di filtraggio dello spam, è fondamentale che le email classificate come "spam" siano effettivamente spam. Un'alta precisione è importante perché non vogliamo che email legittime (non spam) vengano erroneamente contrassegnate come spam (falsi positivi). In questo caso, se il modello ha bassa precisione, potremmo perdere messaggi importanti.
# - In un sistema che rileva una malattia pericolosa, il richiamo è più importante perché è cruciale individuare tutti i pazienti malati, anche a costo di avere alcuni falsi positivi. Se il modello ha un basso richiamo, rischieremmo di non diagnosticare correttamente alcune persone malate (falsi negativi).
#
# ### Con quali dati?
# La scelta dei dati da utilizzare per valutare le performance del modello è un aspetto a cui prestare attenzione. Intuitivamente, vorremmo utilizzare gli stessi dati che abbiamo usato per allenare il modello, ossia quelli appartenenti al _**train set**_ $S$.
# Vediamo come mai questa sia in realtà una cattiva idea.
#
# Prendiamo in esame un semplice problema di regressione. Ogni esempio è costituito da due valori reali, $x$ (l'istanza) e $y$ (l'etichetta) e dunque individua un punto nel piano. I punti neri costituiscono $S$, mentre quelli bianchi corrispondono a dati nuovi, mai visti dal modello, e quindi ancora da etichettare. L'obiettivo è, come sempre, approssimare la ground truth, e questo problema permette di visualizzarlo facilmente. Di fatto, in questo caso, il modello non è altro che una curva, e lo valuteremo in base a quanto si discosta mediamente dai punti del dataset (MSE).
#
# <div style="text-align: center;">
#     <img src="images/regressione.jpeg" alt="Descrizione" width="300" height="200">
# </div>
#
# Chiaramente il modello ideale sarebbe $A$, che interpola bene tutti i punti, sia quelli incontrati in $S$, sia quelli nuovi. Di certo non potremmo dire che $B$ sia un buon modello: si discosta notevolmente dai dati non incontrati. Eppure, se lo valutassimo solo su $T$, performerebbe come $A$ (se non meglio). $B$ soffre del cosiddetto problema di **overfitting**: si è specializzato solo sul train set, non ha colto la relazione sottostante ai dati e dunque non è in grado di generalizzare su dati mai visti.  
# Pertanto, **è fondamentale testare il modello su un _test set_ $T$**, cioè un insieme di dati disgiunto da $S$.
#
# Questo non significa che valutare un modello sul train set sia inutile: performance scarse sul train set indicano che il modello utilizzato non è abbastanza espressivo/potente per il dato problema (fenomeno noto come **underfitting**), e dunque costituiscono una prima informazione preziosa. L’importante è riconoscere che performance buone sul train set non implicano qualità del modello.
# -

# ## Due tipologie di modello
# Presentiamo molto brevemente due tra le numerose tipologie di modello di ML, anche per illustrare più facilmente alcuni concetti più avanti.
#
# ### Albero di decisione
# Si tratta di un albero in cui ogni nodo interno rappresenta una decisione basata su una caratteristica (o attributo) dell'esempio, e le foglie rappresentano le classi o le previsioni finali. Fissata un’altezza massima, l’algoritmo di apprendimento produce, partendo dai dati, un albero di decisione; deve quindi stabilirne la struttura, le condizioni presenti nei nodi interni e le classi nelle foglie.  
# <div style="text-align: center;">
#     <img src="images/decision_tree.png" alt="Descrizione" width="400" height="300">
# </div>
#
# ### KNN
# Gli esempi vengono disposti in uno spazio avente una dimensione per ogni possibile caratteristica/attributo delle istanze. Dunque ogni esempio corrisponde ad un punto in questo spazio.  
# Per classificare una nuova istanza, rappresentata da un nuovo punto, KNN calcola la distanza tra quest'ultimo e tutti i punti del dataset (solitamente utilizzando la distanza euclidea).
# Vengono poi selezionati i $k$ punti più vicini (da qui "$k$-nearest neighbors").
# La classe del nuovo punto viene determinata dalla classe più frequente tra i $k$ vicini selezionati nel caso della classificazione, oppure dalla media dei valori nel caso della regressione.  
# Nel caso di KNN non c’è una vera e propria fase di allenamento (vedremo poi nel paragrafo dedicato a parametri e iperparametri); semplicemente i dati vengono memorizzati.
# <div style="text-align: center;">
#     <img src="images/knn.png" alt="Descrizione" width="300" height="200">
# </div>

# ## Parametri e iperparametri
# I parametri sono le caratteristiche del modello che vengono **definite nella fase di apprendimento**. Gli iperparametri, invece, sono le caratteristiche del modello che devono essere fissate __*prima* della fase di apprendimento__.
#
# Per gli alberi di decisione, ha senso stabilire come prima cosa un’altezza massima, che costituisce quindi un iperparametro. Fissata quella, l’allenamento stabilisce la topologia, le decisioni presenti nei nodi interni e le classi delle foglie; che sono i parametri. **Lo scopo dell'allenamento è trovare i valori ottimali per i parametri del modello**.  
# Prima dell'allenamento, i parametri hanno valori di default o addirittura casuali; al termine dell'allenamento, i parametri hanno i valori ottimali, ossia quelli che consentono al modello di compiere predizioni accurate.  
# Nel caso di KNN, non c’è una vera e propria fase di apprendimento; di fatto non ci sono parametri da stabilire. C’è solo un iperparametro, ossia $k$.
#
# Ma se l’algoritmo di apprendimento trova i parametri, come si trovano gli iperparametri?
#
# ### Tuning degli iperparametri
# In questa sezione prendiamo come riferimento KNN, che ha un solo iperparametro: $k$. Sappiamo che non c’è un vero e proprio allenamento, ma lo includeremo tra le fasi, generalizzando (si può pensare ad una fase di “collocazione dei dati nello spazio”).
#
# Non c’è un modo particolarmente furbo di trovare il valore ottimale di $k$. La cosa migliore da fare è anche la più intuitiva: stabilire un insieme di $n$ possibili valori per $k$ e lanciare un algoritmo di apprendimento per ciascuno di questi valori, generando così $n$ modelli allenati sullo stesso train set $S$. Ciascun modello viene poi valutato e si sceglie quello con la performance migliore.  
# Come già discusso [qua](#Valutare-un-modello), non possiamo misurare le performance limitandoci al $S$. Dunque, per scegliere il modello migliore tra gli $n$ generati, ricorriamo ad un cosiddetto **validation set** $V$, disgiunto da $S$. Denotiamo come $k^*$ l'iperparametro del modello che performa meglio.  
# Viene poi generato un nuovo modello $m$ avente $k = k^*$, ma allenandolo sull'unione $S \cup V$. Chiaramente più dati si danno in pasto al modello, meglio performerà.  
# Infine, si utilizza un apposito test set $T$ per valutare le performance di $m$. Se sono soddisfacenti, si fa un altro “giro” di allenamento, questa volta su $S \cup V \cup T$.
#
# Se si hanno due o più iperparametri, si procede analogamente, testando ogni loro possibile combinazione e scegliendo quella che dà luogo alla performance migliore su $V$. Chiaramente le risorse computazionali richieste aumentano non poco.

# ## Gestire il dataset
# Dato il dataset $ D = \{(x_1, y_1), \dots, (x_n, y_n)\} $, come possiamo ricavarci un train set $S$, un test set $T$ e, se necessario, un validation set $V$?
#
# Di seguito illustreremo alcune tecniche che consentono di farlo, insieme al concetto di campionamento stratificato, fondamentale per la buona riuscita dell'allenamento e della valutazione di un modello.

# ### Campionamento stratificato
# Un dataset non è altro che un campione della popolazione, dunque deve rifletterne la distribuzione. È importante che tale distribuzione sia preservata anche in qualsiasi altro sottoinsieme di $D$ ($S$, $T$ e $V$). Questo lo scopo delle tecniche di _stratified sampling_ (campionamento stratificato).  
# Se $S$ e $T$ avessero distribuzioni diverse, vorrebbe dire testare il modello su una popolazione distribuita diversamente rispetto a quella su cui si è allenato.
#
# Prendiamo come esempio un problema di classificazione binaria. Supponiamo di avere $D$ contenente 500 esempi positivi e 500 esempi negativi, che vogliamo suddividere facendo holdout (link al paragrafo) in un train set $S$ con il 70% degli esempi e un test set $T$ con il 30% degli esempi. In tal caso, un metodo di campionamento stratificato garantirà che $S$ contenga 350 esempi positivi e 350 esempi negativi, e che $T$ contenga 150 esempi positivi e 150 esempi negativi.
#
# Da qua in poi assumeremo sempre che qualsiasi sottoinsieme ricavato da $D$ sia correttamente stratificato.

# ### Holdout
#
# La metodologia Hold Out consiste nel suddividere $ D $ in nel train set $ S $ e nel test set $ T $, in modo tale che $ D = T \cup S $ e $ T \cap S = \emptyset $.
# <div style="text-align: center;">
#     <img src="images/holdout.webp" alt="Descrizione" width="300" height="200">
# </div>
#
# La domanda che sorge è: quanti esempi inserire in $S$? Quanti in $T$? 
# Se insieriamo quasi tutti gli esempi in $S$, allora il modello che otteniamo è un'ottima approssimazione del modello allenato su $D$, il che è desiderabile poiché, generamente, più dati fornisco all'algoritmo di apprendimento, migliore è il modello che ottengo. Tuttavia, la valutazione delle performance è meno affidabile a causa delle dimensioni ridotte di $T$. D'altra parte, se inseriamo più esempi in $T$, la differenza tra il modello addestrato su $S$ e quello addestrato su $D$ diventa eccessiva. Non esiste una soluzione perfetta a questo dilemma; è necessario un compromesso. Una pratica comune è utilizzare dai 2/3 ai 4/5 degli esempi in $D$ per l'addestramento e il resto per il test.
#
# #### Holdout ripetuto
# Poniamo di voler ricavare solo $S$ e $T$. Mantenendo costante il numero di esempi in $S$ (e quindi il numero di esempi in $T$), a partire dallo stesso $D$ si possono ottenere più coppie $(S,T)$. Di fatto, nella pratica, si eseguono $n$ holdout (ad es. 100). Dunque alleno e valuto $n$ modelli, ciascuno su una coppia $(S,T)$ diversa, sempre ottenuta da $D$.
# Come valutazione complessiva, si prende la media delle $n$ valutazioni eseguite, e si allena un modello sull’intero $D$, che sarà poi il modello definitivo.

# ### Cross-validation
#
# La metodologia cross-validation prevede di suddividere $ D $ in $ k $ sottoinsiemi di cardinalità uguale o simile, tali che $ D = \bigcup_{i=1}^{k} D_i $ e $ \forall i, j \in \{1, \dots, k\}, D_i \cap D_j = \emptyset $, con $ i \neq j $. In altre parole, i $k$ sottoinsiemi costituiscono una partizione di $D$.  
# Vengono svolte $ k $ iterazioni; a ogni iterazione, un singolo insieme $ D_i $ non ancora utilizzato come test set, viene scelto come tale, mentre il training set viene composto dall'unione di tutti gli altri sottoinsiemi.  
# <div style="text-align: center;">
#     <img src="images/crossvalidation.png" alt="Descrizione" width="500" height="500">
# </div>
# Finite le iterazioni, disponiamo di $ k $ valutazioni, da cui otteniamo un'unica valutazione complessiva calcolando la media.  
# Poiché la stabilità e l'accuratezza della cross-validation dipendono in gran parte dal valore di $ k $, essa è anche conosciuta come k-fold cross-validation. Il valore di $ k $ più comunemente utilizzato è 10, e il metodo corrispondente è chiamato 10-fold cross-validation.
#
# Come nel metodo holdout, ci sono diversi modi per suddividere il dataset $ D $ in $ k $ sottoinsiemi. Per ridurre l'errore introdotto dalla suddivisione, spesso ripetiamo la suddivisione casuale $ p $ volte e facciamo la media dei risultati delle $ p $ iterazioni di k-fold cross-validation. Un caso comune è la 10-times 10-fold cross-validation.
#
# Un caso speciale di cross-validation è il **Leave-One-Out** (LOO), in cui abbiamo un dataset con cardinalità $ m $ e impostiamo $ k = m $. Le valutazioni sono molto accurate, ma il costo computazionale dell'addestramento di $ m $ modelli può essere proibitivo per dataset di grandi dimensioni.

# ### Ricavare $V$
#
# Finora abbiamo visto come si possono ricavare $S$ e $T$ da $D$, ma non abbiamo discusso il caso in cui si debba anche fare tuning degli iperparametri, ricorrendo quindi al validation set nella modalità descritta nel paragrafo appena citato.   
# In questo caso, le suddivisioni vengono annidate: abbiamo una suddivisione esterna, da $D$ a $S$ e $T$, e una interna, da $S$ a $S$ ridotto e $V$.  
# Le due suddivisioni possono essere eseguite con lo stesso metodo (ad esempio holdout-holdout), oppure con metodi differenti (ad esempio holdout-cv). L'immagine visualizza una suddivisione annidata holdout-cv.
#
# <div style="text-align: center;">
#     <img src="images/tuning.png" alt="Descrizione" width="400" height="400">
# </div>
#
# Merita particolare attenzione il caso in cui si scelga di utilizzare la cross-validation per la suddivisione esterna. A ogni iterazione esterna corrisponde un train set diverso, dunque un modello allenato diversamente, i cui iperparametri ottimali possono quindi differire da quelli dei modelli relativi alle altre iterazioni esterne. Ogni iterazione esterna può potenzialmente produrre un modello "unico" nei suoi iperparametri. In questo caso, possiamo scegliere uno qualsiasi dei modelli addestrati.

# ## Esempi pratici con scikit-learn
# Vediamo degli esempi pratici che coinvolgono KNN. Utilizziamo il **dataset _iris_**, già disponibile in scikit-learn, che comprende osservazioni relative a tre sottospecie di pianta iris: _Setosa_, _Versicolor_ e _Virginica_.  
# Affrontiamo un semplice problema di classificazione binaria: dati i quattro attributi del dataset, dobbiamo stabilire se una pianta appartenga o meno alla sottospecie Setosa.

# +
import numpy as np
from sklearn import datasets
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

iris = datasets.load_iris()
# -

# I quattro attributi che compongono ciascuna istanza si riferiscono a lunghezza e larghezza dei sepali e dei petali delle piante osservate.

iris['feature_names']

# Modifichiamo il dataset perché il target sia di tipo binario. Dobbiamo distinguere solo tra Setosa e non.

iris['target'][iris['target'] == 2] = 1
iris['target_names'] = np.array(['setosa', 'versicolor/virginica'])
iris['target']

# Ora che il dataset è pronto, passiamo alla fase di training. Vogliamo anche fare il tuning dell'iperparametro $k$. Come già discusso, possiamo adottare diversi metodi per suddividere il dataset negli insiemi necessari. Di seguito, vediamo sia un approccio holdout-holdout, che un approccio holdout-cv.

# ### Approccio holdout-holdout
# Eseguiamo il primo holdout per ottenere train set e test set, utilizzando il metodo `train_test_split` di scikit-learn. Attraverso il parametro `test_size`, impostiamo la percentuale di esempi da includere nel test set (e quindi quella da includere nel train set).

# +
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

X_trainval, X_test, y_trainval, y_test = train_test_split(
    iris['data'], 
    iris['target'], 
    test_size=0.3,  
    stratify=iris['target']
)
# -

# Ripetiamo l'holdout per ottenere, a partire dal train set appena ricavato, un train set ridotto e un validation set.  Questa volta impostiamo `test_size` a 0.43, per far sì che train set e validation set abbiano (circa) la stessa dimensione.

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval,
    y_trainval,
    test_size=0.43,
    stratify=y_trainval
)

# Ora diamo il via ad una classica ricerca del massimo. Ad ogni iterazione, inizializziamo un modello con un $k$ diverso (nel range specificato), lo alleniamo sul train set e ne calcoliamo l'accuratezza sul validation set. Se questa è più grande della migliore corrente, aggiorniamo il modello migliore e la sua accuratezza.
# Al termine del ciclo, la variabile `best_model` conterrà il modello avente il $k$ ottimale.  

# +
best_model = None
best_accuracy = 0
        
for k in range(1, 10, 2):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_val, model.predict(X_val))
    if(accuracy > best_accuracy):
        best_model = model
        best_accuracy = accuracy
# -

# Come già discusso, è solito fare un cosiddetto _**refit**_, ossia un altro "giro" di addestramento utilizzando anche i dati con cui si è appena valutato il modello (in questo caso, il validation set). Chiaramente addestriamo un modello avente il $k$ ottimale, ossia quello di `best_model`.  
# Valutiamo poi tale modello sul test set.

# +
model = KNeighborsClassifier(best_model.n_neighbors)
model.fit(X_trainval, y_trainval)

# Valutazione finale su test set
accuracy_score(y_test, best_model.predict(X_test))
# -

# Eseguiamo un ultimo refit, questa volta sull'intero dataset.

model = KNeighborsClassifier(best_model.n_neighbors)
model.fit(iris['data'], iris['target'])

# ### Approccio holdout-cv
# Il primo holdout è analogo a quello della sezione precedente.

# +
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

X_trainval, X_test, y_trainval, y_test = train_test_split(
    iris['data'], 
    iris['target'], 
    test_size=0.3,  
    stratify=iris['target']
)
# -

# Utilizziamo un oggetto della classe `GridSearchCV`, il cui metodo `fit` automaticamente, oltre ad allenare il modello, svolge il tuning degli iperparametri (in questo caso cerchiamo solo $k$) tramite cross-validation.  
# Al costruttore di `GridSearchCV` passiamo:
# - il modello che vogliamo allenare
# - un dizionario i cui elementi sono del tipo _'nome iperparametro': lista di possibili valori_
# - numero di fold (`cv`)

# +
model = KNeighborsClassifier()
parameters = {'n_neighbors': [1, 3, 5, 7, 9]}
gs = GridSearchCV(model, parameters, cv=5)
gs.fit(X_trainval, y_trainval)

# Valutazione su test set
accuracy_score(y_test, gs.predict(X_test))
# -

# Per finire, un refit. Da notare che, di default, il primo refit (quello fatto su tutto il set _trainval_) viene automaticamente svolto dal metodo `fit` di `GridSearchCV`.

model = KNeighborsClassifier(gs.best_estimator_.n_neighbors)
model.fit(iris['data'], iris['target'])

# ### Approcio cv-cv
#
# Utilizziamo il metodo `cross_val_score`, che restituisce un array di valutazioni (`cv_scores`); una per ciascuna iterazione di cross-validation. Passiamo come parametri:
# - il modello da allenare e valutare ad ogni iterazione. Da notare che forniamo un oggetto `GridSearchCV`, il cui metodo `fit` svolge il tuning degli iperparametri tramite cross-validation (vedi paragrafo precedente)
# - il dataset, composto da osservazioni ed etichette
# - il numero di *fold* (parametro `cv`), che quindi è anche il numero di valutazioni che verranno inserite nell'array restituito
#
# La cross-validation interna è quindi svolta dal metodo `fit` di `GridSearchCV`, mentre quella esterna da `cross_val_score`.  
# Per ottenere una stima finale, calcoliamo la media delle valutazioni restituite da `cross_val_score`.

# +
from sklearn.model_selection import cross_val_score

model = KNeighborsClassifier()
parameters = {'n_neighbors': [1, 3, 5, 7, 9]}
gs = GridSearchCV(model, parameters, cv=5)
cv_scores = cross_val_score(gs, iris['data'], iris['target'], cv=5)
np.mean(cv_scores)
# -
# Come al solito, eseguiamo un refit sull'intero dataset per ottenere il modello finale.  
# NB: se non viene specificato altrimenti nel costruttore di `GridSearchCV`, il metodo `fit`, dopo aver trovato il miglior iperparametro, esegue un refit su tutto il dataset passatogli in input.

# +
gs.fit(iris['data'], iris['target'])

# Modello finale
gs.best_estimator_
# -

# ### Approcio generalizzato
#
# L'idea è arrivare ad una funzione generalizzata, in grado di svolgere training e testing di un modello, con anche tuning degli iperparametri, secondo diverse metodologie.
#
# Definiamo prima alcune funzioni ausiliarie. La prima è `make_hp_configurations`, che prende in input una griglia di iperparametri (come quella vista per `GridSearchCV`) e restituisce una lista contenente tutte le possibili combinazioni dei valori degli stessi.

# +
import itertools as it

def make_hp_configurations(grid):
    """
    Genera tutte le combinazioni possibili di iperparametri basate su una griglia fornita.
    
    Parameters:
    ----------
    grid : dict
        Griglia degli iperparametri. Ogni coppia k-v è del tipo `nome_iperparametro : lista_possibili_valori`.
    
    Returns:
    -------
    list of dict
        Tutte le possibili configurazioni per gli iperparametri.
    
    Examples:
    --------
    >>> grid = {'n_neighbors': [1, 3], 'metric': ["nan_euclidean", "manhattan"]}
    >>> make_hp_configurations(grid)
    [{'n_neighbors': 1, 'metric': "nan_euclidean"},
     {'n_neighbors': 1, 'metric': "manhattan"},
     {'n_neighbors': 3, 'metric': "nan_euclidean"},
     {'n_neighbors': 3, 'metric': "manhattan"}]
    """
    return [{n: v for n, v in zip(grid. keys(), t)} for t in it.product(*grid.values())]


# -

# La seconda funzione ausiliaria è `fit_estimator`, che riceve un modello, una configurazione per i suoi iperparametri, ed esegue il training sul dataset specificato.

def fit_estimator(X, y, estimator, hp_conf):
    """
    Configura e addestra un modello predittivo con gli iperparametri forniti.

    Parameters:
    ----------
    X : array-like
        Osservazioni del dataset.
        
    y : array-like
        Etichette del dataset.
    
    estimator : oggetto estimator
        Modello predittivo che implementa i metodi `set_params` e `fit`.
    
    hp_conf : dict
        Dizionario contenente i nomi degli iperparametri e i rispettivi valori.
    
    Returns:
    --------
    Nessuno
        La funzione addestra l'estimatore in loco.

    Esempio:
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> X = [[1, 2], [3, 4], [5, 6]]
    >>> y = [0, 1, 0]
    >>> estimator = RandomForestClassifier()
    >>> hp_conf = {'n_estimators': 100, 'max_depth': 5}
    >>> fit_estimator(X, y, estimator, hp_conf)
    """
    
    estimator.set_params(**hp_conf)
    estimator.fit(X, y)


# La funzione `get_score` riceve un modello e una funzione che calcola una specifica metrica, e restituisce il valore calcolato per la stessa sul dataset specificato.  
# NB: le funzioni sono _cittadine di prima classe_ in python, ecco perché possiamo utilizzarne una come parametro in questo modo.  
# Per le metriche utilizziamo il modulo `metrics` di scikit-learn, che fornisce una serie di funzioni apposite.

# +
import sklearn.metrics as metrics

def get_score(X_test, y_test, estimator, scorer): 
    """
    Calcola il punteggio del modello predittivo usando il metodo di scoring specificato.
    
    Parameters:
    ----------
    X_test : array-like
        Osservazioni del dataset da classificare.
        
    y_test : array-like
        Etichette del dataset con sui confrontare le predizioni.
        
    estimator : estimator object
        Modello predittivo che implementa la funzione `predict()`.
        
    scorer : callable
        Funzione di scoring che, date le etichette e le predizioni, restituisce un punteggio.
    
    Returns:
    -------
    float
        Il punteggio calcolato in base alle previsioni fatte dal modello sui dati di test e
        valutato con la funzione di scoring.

    Example:
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.metrics import accuracy_score
    >>> X_train = [[1, 2], [3, 4], [5, 6], [7, 8]]
    >>> y_train = [0, 1, 0, 1]
    >>> X_test = [[9, 10], [11, 12]]
    >>> y_test = [1, 0]
    >>> estimator = RandomForestClassifier()
    >>> estimator.fit(X_train, y_train)
    >>> score = get_score(X_test, y_test, estimator, accuracy_score)
    >>> print(score)
    """
    
    return scorer(y_test, estimator.predict(X_test))


# -

# La funzione `check_best` riceve il valore di due metriche: la prima rappresenta quella appena calcolata e la seconda quella migliore trovata. Viene restituito `True` se `score` è meglio di `best_score`, dove "meglio" dipende dal booleano minimize, impostato a `True` se la metrica è da minimizzare e a `False` altrimenti.

def check_best(minimize, score, best_score):
    """
    Verifica se il punteggio corrente è migliore rispetto al miglior punteggio finora, 
    in base alla strategia di minimizzazione o massimizzazione.
    
    Parameters:
    ----------
    minimize : bool
        Indica se la strategia di ottimizzazione è di minimizzazione (`True`) o di massimizzazione (`False`).
        
    score : float
        Punteggio corrente che si vuole confrontare con il miglior punteggio.
        
    best_score : float
        Miglior punteggio trovato finora.
    
    Returns:
    -------
    bool
        `True` se il punteggio corrente è migliore del miglior punteggio in base alla strategia 
        di ottimizzazione; `False` altrimenti.
    
    Examples:
    --------
    >>> check_best(True, 0.2, 0.5)
    True
    >>> check_best(False, 0.7, 0.5)
    True
    >>> check_best(True, 0.6, 0.5)
    False
    """
    
    return (minimize and score < best_score) or (not minimize and score > best_score)


# #### Funzione `learn`
# Questa funzione è generalizzata rispetto a dataset, modello, relativi iperparametri da modulare, metodologia con cui si suddivide il dataset per la valutazione (cv, holdout,...) e metriche di valutazione, sia sul test set che sul validation set.  
# Di fatto, i parametri sono:
# - osservazioni (`X`) ed etichette (`y`) del dataset
# - modello (`estimator`)
# - griglia degli iperparametri (`param_grid`)
# - metodologie di suddivisione del dataset per la valutazione (`outer_split_method` e `inner_split_method`; spiegazioni approfondite seguono la funzione)
# - metriche di valutazione (`val_scorer`, `test_scorer`) e relativa indicazione sulla massimizzazione o minimizzazione (`minimize_val_scorer`, `minimize_test_scorer`); di default, la metrica di valutazione è la radice dell'errore quadratico medio (che va quindi minimizzato), sia per testing che per validation.

def learn(X, y, estimator, param_grid, outer_split_method, inner_split_method,
            val_scorer=metrics.root_mean_squared_error, minimize_val_scorer=True, 
            test_scorer=metrics.root_mean_squared_error, minimize_test_scorer=True):
    """
    Addestra un modello predittivo e ottimizza i suoi iperparametri usando una procedura generica 
    di divisione dei dati (es: cross-validation, hold-out).
    
    Parameters:
    ----------
    X : array-like
        Osservazioni del dataset.
        
    y : array-like
        Etichette del dataset.
        
    estimator : estimator object
        Modello predittivo che implementa i metodi `fit()`, set_params() e `predict()`.
        
    param_grid : dict
        Griglia degli iperparametri. Ogni coppia k-v è del tipo `nome_iperparametro : lista_possibili_valori`.
        
    outer_split_method : object
        Oggetto dotato del metodo `split()`, che genera la suddivisione esterna 
        dei dati in sottoinsiemi di training/validation e test.
        
    inner_split_method : object
        Oggetto dotato del metodo `split()`, che genera la suddivisione interna 
        dei dati in sottoinsiemi di training e validation.
        
    val_scorer : callable, optional, default=metrics.root_mean_squared_error
        Funzione di scoring per valutare le prestazioni del modello sul validation set.
        
    minimize_val_scorer : bool, optional, default=True
        Se `True`, la funzione cerca di minimizzare la metrica calcolata `val_scorer`. 
        Se `False`, cerca di massimizzarlo.
        
    test_scorer : callable, optional, default=metrics.root_mean_squared_error
        Funzione di scoring per valutare le prestazioni del modello sul test set.
        
    minimize_test_scorer : bool, optional, default=True
        Se `True`, la funzione cerca di minimizzare la metrica calcolata da `test_scorer`. 
        Se `False`, cerca di massimizzarlo.
    
    Returns:
    -------
    estimator object
        Il modello addestrato sull'intero dataset con la configurazione di iperparametri ottimale.
        
    float
        Lo score finale del modello.
    """

    outer_scores = []

    best_score = np.inf if minimize_val_scorer else -np.inf
    best_conf = None

    for trainval_index, test_index in outer_split_method.split(X, y):
        # Ad ogni iterazione corrisponde una suddivisione diversa in trainval e test
        
        X_trainval, X_test = X[trainval_index], X[test_index]
        y_trainval, y_test = y[trainval_index], y[test_index]
        
        best_inner_score = np.inf if minimize_test_scorer else -np.inf
        best_inner_conf = None
        
        for hp_conf in make_hp_configurations(param_grid):
            # Ad ogni iterazione corrisponde una configurazione diversa degli iperparametri 
            conf_scores = []
            
            for train_index, val_index in inner_split_method.split(X_trainval, y_trainval):
                # Ad ogni iterazione corrisponde una suddivisione diversa in train e val
                
                X_train, X_val = X_trainval[train_index], X_trainval[val_index]
                y_train, y_val = y_trainval[train_index], y_trainval[val_index]

                fit_estimator(X_train, y_train, estimator, hp_conf)
                conf_scores.append(get_score(X_val, y_val, estimator, val_scorer))

            conf_score = np.mean(conf_scores)

            if check_best(minimize_val_scorer, conf_score, best_inner_score):
                best_inner_score, best_inner_conf = conf_score, hp_conf

        fit_estimator(X_trainval, y_trainval, estimator, best_inner_conf)
        outer_score = get_score(X_test, y_test, estimator, test_scorer)
        outer_scores.append(outer_score)

        if check_best(minimize_test_scorer, outer_score, best_score):
            best_score, best_conf = outer_score, best_inner_conf

    fit_estimator(X, y, estimator, best_conf)
    return estimator, np.mean(outer_scores)


# Di particolare importanza sono i parametri `outer_split_method` e `inner_split_method`. Al loro contenuto è richiesto un metodo *generatore* `split(X, y)`, che utilizziamo per ottenere una partizione del dataset (il quale è sottoforma di `X` e `y`) composta da due insiemi.
# Essendo `split` un generatore, **restituisce ogni volta una coppia diversa di insiemi di indici** per il dataset specificato: uno per gli elementi del primo insieme ed uno per quelli del secondo. Dunque,  utilizzando la funzione `enumerate()`, possiamo iterare sulle coppie generate da `split` per realizzare, ad esempio, una cross-validation.  
# Noi passeremo come valori per `outer_split_method` e `inner_split_method` degli oggetti appartenenti a classi come `StratifiedKFold` oppure `StratifiedShuffleSplit`, i cui metodi `split` permettono di ottenere rispettivamente una divisione secondo cross-validation e una divisione secondo holdout ripetuto. Esempi semplici (ma chiari) del funzionamento di `split` in entrambi i casi si trovano sulla documentazione.
#
# Ecco un semplice esempio di utilizzo di `learn`, sempre sul problema relativo al dataset iris. Vogliamo allenare e valutare un modello KNN, trovando anche l'iperparametro migliore. Scegliamo di utilizzare un approccio cv-holdoutRipetuto, sfruttando le classi sopra menzionate.  

# +
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit

folds = 5
skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
sss = StratifiedShuffleSplit(n_splits=5, test_size=1/(folds-1), random_state=42)
X = iris['data']
y = iris['target']
hp_grid = {'n_neighbors': [1, 3, 5, 7, 9]}
model = KNeighborsClassifier()

learn(X, y, model, hp_grid, skf, sss, 
    val_scorer=metrics.accuracy_score, minimize_val_scorer=False,
    test_scorer=metrics.accuracy_score, minimize_test_scorer=False)


# -

# Il parametro `test_size` del costruttore di `StratifiedShuffleSplit` è impostato in modo tale da avere sempre test set e validation set della stessa dimensione (circa).  
# Il parametro `random_state` serve per rendere i risultati riproducibili.  
# Da notare anche che scegliamo di utilizzare l'accuratezza come metrica di valutazione.

# #### Parallelizzare

# Possiamo modificare la funzione `learn` per fare in modo che la fase di tuning degli iperparametri sia svolta in parallelo, **testando contemporaneamente più configurazioni diverse**, quindi riducendo il tempo di esecuzione.  
# Isoliamo proprio la parte di allenamento e valutazione della singola configurazione in una funzione apposita, che chiamiamo `fit_and_score`.

def fit_and_score(estimator,
                  X, y,
                  hp_conf, split_method,
                  scorer=metrics.root_mean_squared_error):
    """
    Addestra il modello con una specifica configurazione di iperparametri; lo valuta con 
    la metrica e la metodologia di suddivisione del dataset fornite.
    
    Parameters:
    ----------
    estimator : estimator object
        Modello predittivo da addestrare. Deve implementare i metodi `fit()` e `set_params()`.
        
    X : array-like
        Osservazioni del dataset.
        
    y : array-like
        Etichette del dataset.
        
    hp_conf : dict
        Dizionario contenente i nomi degli iperparametri e i rispettivi valori.
        
    split_method : object
        Oggetto dotato del metodo `split()`, che genera la suddivisione 
        dei dati in sottoinsiemi di training e testing.
        
    scorer : callable, optional, default=metrics.root_mean_squared_error
        Funzione di scoring per valutare le prestazioni del modello.
    
    Returns:
    -------
    float
        Score finale ottenuto dal modello.
    dict
        Configurazione di iperparametri utilizzata per addestrare il modello.
    """
    
    estimator.set_params(**hp_conf)
    conf_scores = []
            
    for train_index, val_index in split_method.split(X, y):
                
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        estimator.fit(X_train, y_train)
        conf_scores.append(get_score(X_val, y_val, estimator, scorer))

    return np.mean(conf_scores), hp_conf


# Possiamo ora definire la funzione `learn_parallel`.

# +
from joblib import Parallel, delayed
import copy

def learn_parallel(X, y, estimator, param_grid, outer_split_method, inner_split_method, n_jobs,
            val_scorer=metrics.root_mean_squared_error, minimize_val_scorer=True, 
            test_scorer=metrics.root_mean_squared_error, minimize_test_scorer=True):
    """
    Addestra un modello predittivo e ottimizza i suoi iperparametri usando una procedura generica 
    di divisione dei dati (es: cross-validation, hold-out).
    
    Parameters:
    ----------
    X : array-like
        Osservazioni del dataset.
        
    y : array-like
        Etichette del dataset.
        
    estimator : estimator object
        Modello predittivo che implementa i metodi `fit()`, set_params() e `predict()`.
        
    param_grid : dict
        Griglia degli iperparametri. Ogni coppia k-v è del tipo `nome_iperparametro : lista_possibili_valori`.
        
    outer_split_method : object
        Oggetto dotato del metodo `split()`, che genera la suddivisione esterna 
        dei dati in sottoinsiemi di training/validation e test.
        
    inner_split_method : object
        Oggetto dotato del metodo `split()`, che genera la suddivisione interna 
        dei dati in sottoinsiemi di training e validation.

    n_jobs: int
        Numero di processi da eseguire in parallelo.
        
    val_scorer : callable, optional, default=metrics.root_mean_squared_error
        Funzione di scoring per valutare le prestazioni del modello sul validation set.
        
    minimize_val_scorer : bool, optional, default=True
        Se `True`, la funzione cerca di minimizzare la metrica calcolata `val_scorer`. 
        Se `False`, cerca di massimizzarlo.
        
    test_scorer : callable, optional, default=metrics.root_mean_squared_error
        Funzione di scoring per valutare le prestazioni del modello sul test set.
        
    minimize_test_scorer : bool, optional, default=True
        Se `True`, la funzione cerca di minimizzare la metrica calcolata da `test_scorer`. 
        Se `False`, cerca di massimizzarlo.
    
    Returns:
    -------
    estimator object
        Il modello addestrato sull'intero dataset con la configurazione di iperparametri ottimale.
        
    float
        Lo score finale del modello.
    """

    outer_scores = []

    best_score = np.inf if minimize_val_scorer else -np.inf
    best_conf = None

    for trainval_index, test_index in outer_split_method.split(X, y):
        
        X_trainval, X_test = X[trainval_index], X[test_index]
        y_trainval, y_test = y[trainval_index], y[test_index]
        
        best_inner_score = np.inf if minimize_test_scorer else -np.inf
        best_inner_conf = None
        
        results = Parallel(n_jobs=n_jobs)(delayed(fit_and_score)(copy.deepcopy(estimator),
                                                              X_trainval, y_trainval,
                                                              hp_conf, inner_split_method,
                                                              scorer=val_scorer)
                                       for hp_conf in make_hp_configurations(param_grid))
        
        result = sorted(results, key=lambda t: t[0])[0]
        if check_best(minimize_val_scorer, result[0], best_inner_score):
                best_inner_score, best_inner_conf = result[0], result[1]

        fit_estimator(X_trainval, y_trainval, estimator, best_inner_conf)
        outer_score = get_score(X_test, y_test, estimator, test_scorer)
        outer_scores.append(outer_score)

        if check_best(minimize_test_scorer, outer_score, best_score):
            best_score, best_conf = outer_score, best_inner_conf

    fit_estimator(X, y, estimator, best_conf)
    return estimator, np.mean(outer_scores)


# -

# L'esecuzione in parallelo è merito di questa porzione di codice:
#
# ```python
# Parallel(n_jobs=n_jobs)(delayed(fit_and_score)(copy.deepcopy(estimator),
#                                                               X_trainval, y_trainval,
#                                                               hp_conf, inner_split_method,
#                                                               scorer=val_scorer)
#                                        for hp_conf in make_hp_configurations(param_grid))
# ```
# Il costruttore `Parallel(n_jobs=n_jobs)` restituisce un oggetto di tipo `Parallel`, che può essere chiamato esattamente come una funzione (vedi oggetti callable). Noi lo invochiamo passandogli come argomento la funzione che vogliamo venga eseguita in parallelo, ossia `fit_and_score`. Tuttavia, non lo facciamo direttamente: utilizziamo la funzione `delayed`, che permette di specificare una funzione (coi suoi argomenti) senza effettivamente chiamarla, per fare in modo che la sua esecuzione sia parallela e non sequenziale.  
# L'espressione `for hp_conf in make_hp_configurations(param_grid)` è una **generator expression**, ossia un'espressione che produce un oggetto generatore, che genera i valori uno alla volta al momento della richiesta; in questo caso specifico, una alla volta, genera le configurazioni degli iperparametri.  
# Si usa questo tipo di espressione in modo che Parallel assegni a ogni iterabile (elemento generato) un core diverso.
#
# L'utilizzo di `learn_parallel` è analogo a quello di `learn`, con l'unica differenza data dalla presenza del parametro `n_jobs`. Per stabilire quanti processi possiamo eseguire in parallelo, possiamo importare il modulo **multiprocessing**, la cui funzione `cpu_count()` restituisce il numero di core dell'elaboratore con cui stiamo lavorando.

import multiprocessing as mp
n_jobs = mp.cpu_count()
n_jobs
