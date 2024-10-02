# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
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
# Vengono svolte $ k $ iterazioni; a ogni iterazione, un singolo insieme $ D_i $ non ancora utilizzato come test set, viene scelto come tale, mentre il training set viene composto dall'unione di tutti gli altri sottoinsiemi. A questo punto, possiamo ottenere la valutazione del modello.  
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

# ## Esempi pratici usando scikit-learn
# Vediamo degli esempi pratici che coinvolgono KNN. Utilizziamo il **dataset _iris_**, già disponibile in scikit-learn, che comprende osservazioni relative a tre sottospecie di pianta iris: _Setosa_, _Versicolor_ e _Virginica_.  
# Affrontiamo un semplice problema di classificazione binaria: dati i quattro attributi del dataset, dobbiamo stabilire se una pianta appartenga o meno alla sottospecie Setosa.

# +
import numpy as np
from sklearn import datasets

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
gs = GridSearchCV(model, parameters, cv=10)
gs.fit(X_trainval, y_trainval)

# Valutazione su test set
accuracy_score(y_test, gs.predict(X_test))
# -

# Per finire, un refit. Da notare che, di default, il primo refit (quello fatto su tutto il set _trainval_) viene automaticamente svolto dal metodo `fit` di `GridSearchCV`.

model = KNeighborsClassifier(gs.best_estimator_.n_neighbors)
model.fit(iris['data'], iris['target'])
