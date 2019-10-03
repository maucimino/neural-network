% Il seguente script addestra una rete con l'algoritmo resilient back propagation 
% su un training set/validation set estratti casualmente dal dataset MNIST.
% Tali training e validation sono stati estratti in modo da non contenere
% elementi comuni oppure elementi ripetuti. Lo stesso e' stato fatto per il
% test set, su cui verra' sperimentata la rete addestrata. Infine, verra'
% calcolata l'accuratezza della rete confrontando le risposte con le labels
% effettive dei digits estratti dal training set. Tale configurazione degli
% hyper-parametri ha raggiunto un accuratezza massima del 95%.

% Path alla cartella contenente le funzioni
addpath('./functions/');
% Path alla cartella contenente il dataset MNIST
addpath('./mnist/');

% Numero di nodi interni della rete
HIDDEN_NODES = 320;
% Numero di digits da inserire nel training set
TRAINING_SET_SIZE = 15000;
% Numero di digits da inserire nel validation set
VALIDATION_SET_SIZE = 7500;
% Numero di digits da inserire nel test set
TEST_SET_SIZE = 4000;
% Numero di epoche di addestramento
% Si noti che l'algoritmo di training e' stato implementato per fermarsi
% quando si raggiunge l'overfitting, quindi il numero di epoche potrebbe
% essere minore di quello specificato
EPOCHS = 150;
% Limite inferiore dell'intervallo di valori da generare casualmente nella
% matrice dei pesi al momento della creazione della rete.
NETWORK_INF_WEIGHTS = -0.09;
% Limite superiore dell'intervallo di valori da generare casualmente nella
% matrice dei pesi al momento della creazione della rete.
NETWORK_SUP_WEIGHTS = 0.09;
% Funzione di attivazione dei nodi di output
OUTPUT_ACTIVATION_FUNCTION = @identityFunction;
% Funzione di attivazione dei nodi del layer interno della rete
HIDDEN_ACTIVATION_FUNCTION = @sigmoidFunction;
% Funzione di errore per il training
ERROR_FUNCTION = @crossEntropyFunction;
% Eta- per l'algoritmo resilient back propagation
ETA_MINUS = 0.5;
% Eta+ per l'algoritmo resilient back propagation
ETA_PLUS = 1.2;
% Flag per l'attivazione del softmax sui nodi di output della rete dopo la
% forward propagation
SOFTMAX_FLAG = true;
% Flag per la stampa a video degli errori ottenuti sul training e
% validation set durante ogni epoca
PRINT_ERROR_FLAG = true;

% Estrazione dal dataset MNIST delle digits e delle labels
[digits, labels] = loadMNISTDataset('./mnist/train-images-idx3-ubyte', './mnist/train-labels-idx1-ubyte');

% Estrazione casuale del training, validation e test set dal dataset MNIST
% precedentemente estratto. Tutti i set sono tra loro disgiunti e non
% contengono elementi ripetuti.
[trainingSetData, trainingSetLabels, validationSetData, validationSetLabels, testSetData, testSetLabels] = buildSets(digits, labels, TRAINING_SET_SIZE, VALIDATION_SET_SIZE, TEST_SET_SIZE, true);

% Creazione di una rete neurale feed-forward multi-strato. Il numero di
% nodi di input e' 784 (dimensione di un digits nella matrice delle
% immagini), il numero di nodi di output e' 10 (classificazione in dieci
% classi: da 0 a 9).
[neuralNetwork] = newFFMLNeuralNetwork(size(trainingSetData, 2), 10, OUTPUT_ACTIVATION_FUNCTION, [struct('layerSize', HIDDEN_NODES, 'activationFunction', HIDDEN_ACTIVATION_FUNCTION)], NETWORK_INF_WEIGHTS, NETWORK_SUP_WEIGHTS);

% Per misurare le performance in secondi del training e testing della rete.
tic;

% Training della rete neurale precedentemente creata utilizzando un
% approccio BATCH e l'algoritmo resilient back propagation.
[neuralNetwork, trainingSetErrors, validationSetErrors] = trainNetworkResilientBackPropagation(neuralNetwork, trainingSetData, trainingSetLabels, validationSetData, validationSetLabels, EPOCHS, ERROR_FUNCTION, ETA_MINUS, ETA_PLUS, SOFTMAX_FLAG, PRINT_ERROR_FLAG);

% Forward propagation della rete addestrata utilizzando come input il test
% set.
[neuralNetwork] = forwardPropagation(neuralNetwork, testSetData, SOFTMAX_FLAG);

% Calcolo dell'accuratezza delle risposte della rete, confrontandole con le
% label effettive del test set.
[totalAccuracy] = evaluateNeuralNetworkClassifier(neuralNetwork.z{neuralNetwork.numOfHiddenLayers+1}, testSetLabels);

% Stampa a video dell'accuratezza ottenuta dalla rete sul test set.
fprintf("\nNetwork's accuracy: %d%% \n", int16(totalAccuracy*100));

% Stampa a video del tempo impiegato per il training ed il test della rete.
fprintf("\nTime for training and testing the network: %f seconds \n", toc);

% Creazione e visualizzazione dei grafici che mostrano l'andamento della
% funzione di errore sul training e validation set.
plotErrors(trainingSetErrors, validationSetErrors);
