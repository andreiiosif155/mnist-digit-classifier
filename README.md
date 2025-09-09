MNIST-101

Abstract:
Developed a simple neural network in Octave to classify handwritten digits (0–9) from the MNIST dataset:
- Implemented dataset handling: loading .mat files, random shuffling, and train/test splitting
- Designed weight initialization with symmetric randomization to avoid learning stagnation
- Implemented forward propagation with sigmoid activations and bias handling
- Built the cost function with backpropagation, including gradient computation and L2 regularization
- Implemented prediction function to classify digits based on output layer activations


A more detailed implementation can be found below:
load_dataset
- Se incarca fisierul .mat ce contine datele salvate sub forma unei structuri
  cu campurile X si y cu ajutorul functiei load[1].
- Se extrag separat matricea X (predictorii) si vectorul y (iesirile)
  din structura incarcata.

split_dataset
- Se genereaza o permutare aleatoare a indicilor de la 1 la m cu ajutorul
  functiei randperm[2], pentru a amesteca datele.
- Se aplica permutarea asupra liniilor matricelor X si y pentru a asigura
  o distributie aleatoare a exemplelor.
- Se calculeaza numarul de exemple pentru setul de antrenare ca floor[3](percent * m).
- Se extrag primele m_train linii pentru setul de antrenare (X_train, y_train),
  iar restul pentru testare (X_test, y_test).

initialize_weights
- Se calculeaza valoarea epsilon, conform formulei de initializare simetrica.
- Se initializeaza aleator matricea de greutati cu dimensiunea (L_next x (L_prev + 1))
  folosind functia rand[4].
- Valorile generate sunt distribuite uniform in intervalul [-epsilon, epsilon]
  pentru a preveni simetria in procesul de invatare.

cost_function
- Se reconstruieste structura retelei neuronale pornind de la vectorul 
  parametrilor params, folosind functia reshape[5] pentru a extrage matricile
  Theta1 si Theta2.
- Se realizeaza propagarea inainte aplicand functia sigmoid pe fiecare strat
  si adaugand bias corespunzator.
- Se construieste matricea Y prin selectarea liniilor corespunzatoare etichetelor
  y din eye(output_layer_size), fiecare linie indicand clasa corecta pentru exemplul respectiv.
- Se implementeaza algoritmul de backpropagation: se calculeaza erorile d3 pentru iesire
  si d2 pentru stratul ascuns.
- In calculul lui d2, se elimina prima coloana din produsul d3 * Theta2 deoarece
  corespunde bias-ului, iar derivata functiei sigmoid se aplica doar pe activarea
  fara bias (a2(:,2:end)).
- Se calculeaza matricile Delta1 si Delta2 ca produs al erorilor si activarilor,
  iar apoi se obtin gradientii medii Theta1_grad si Theta2_grad
- Se adauga termenul de regularizare doar pe coloanele diferite de bias (coloana 1).
- Functia de cost J se calculeaza cu entropia incrucisata intre predictii si valorile
  reale cu ajutorul functiei sum[6].

predict_classes
- Se reconstruieste arhitectura retelei neuronale pornind de la vectorul weights,
  folosind functia reshape[5] pentru a obtine matricile Theta1 si Theta2,
  corespunzatoare straturilor ascuns si de iesire.
- Se aplica propagarea inainte asupra datelor de intrare X:
  - Se adauga coloana de bias in a1 pentru stratul de intrare
  - Se calculeaza activarea stratului ascuns a2 aplicand functia sigmoid peste z2 = a1 * Theta1'
  - Se adauga bias si stratului ascuns
  - Se calculeaza activarea finala a3 cu sigmoid(z3), unde z3 = a2 * Theta2' 
  - Se determina clasa prezisa pentru fiecare exemplu alegand pozitia cu valoarea maxima din vectorul a3
    (functia max[7] cu argumentul 2 returneaza indicii pe linie).

load[1] - https://octave.sourceforge.io/octave/function/load.html
randperm[2] - intoarce o permutare aleatoare - https://octave.sourceforge.io/octave/function/randperm.html
floor(x)[3] - intoarce cel mai mare numar intreg mai mic decat x - https://octave.sourceforge.io/octave/function/floor.html
rand[4] - intoarce o matrice aleatoare cu elemente distribuite uniform pe intervalul (0,1) - https://octave.sourceforge.io/octave/function/rand.html
reshape[5] - https://octave.sourceforge.io/octave/function/reshape.html
sum[6] - mai eficient decat a aduna elementele cu un for in MATLAB/Octave
       - comparabil de eficient cu a aduna cu un for in C
       - https://docs.octave.org/v9.1.0/Sums-and-Products.html
max[7] - https://octave.sourceforge.io/octave/function/max.html
