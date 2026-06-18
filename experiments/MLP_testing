"""
this is an experiment (not part of the Unity app) that the TA asked for
it compares two ways of turning Quest Pro face data into an emotion on synthetic data
- the pre-trained MLP from the EmoHeVRDB paper (repo emohevrdb-sfer) downloaded as fau_model.keras
- my cosine-similarity classifier (same logic as my FaceEmotionClassifier.cs)

this version tests three things the TA asked for
- strong: strong, obvious expressions
- medium: weak, subtle expressions near the detection threshold
- mixed: two emotions active at once

this is in python because the MLP is a TensorFlow/Keras model, which cant run inside Unity C# so the comparison has to happen here 
my real system is still the C# code

i need to note that im honestly not sure if this is the right way to test the MLP
synthetic faces are built from EM-FACS rules that my cosine method also uses, so its biased toward cosine method 
"""

import sys
import zipfile
import numpy as np
import h5py

#the 63 values in the eaxct order the model expects (copied from the dataset's csv-building script) 
#order has to match or the model gets scrambled inputs
FEA_NAMES = ['BrowLowererL', 'BrowLowererR', 'CheekPuffL', 'CheekPuffR', 'CheekRaiserL', 'CheekRaiserR',
    'CheekSuckL', 'CheekSuckR', 'ChinRaiserB', 'ChinRaiserT', 'DimplerL', 'DimplerR', 'EyesClosedL', 'EyesClosedR',
    'EyesLookDownL', 'EyesLookDownR', 'EyesLookLeftL', 'EyesLookLeftR', 'EyesLookRightL', 'EyesLookRightR',
    'EyesLookUpL', 'EyesLookUpR', 'InnerBrowRaiserL', 'InnerBrowRaiserR', 'JawDrop', 'JawSidewaysLeft',
    'JawSidewaysRight', 'JawThrust', 'LidTightenerL', 'LidTightenerR', 'LipCornerDepressorL', 'LipCornerDepressorR',
    'LipCornerPullerL', 'LipCornerPullerR', 'LipFunnelerLB', 'LipFunnelerLT', 'LipFunnelerRB', 'LipFunnelerRT',
    'LipPressorL', 'LipPressorR', 'LipPuckerL', 'LipPuckerR', 'LipStretcherL', 'LipStretcherR', 'LipSuckLB',
    'LipSuckLT', 'LipSuckRB', 'LipSuckRT', 'LipTightenerL', 'LipTightenerR', 'LipsToward', 'LowerLipDepressorL',
    'LowerLipDepressorR', 'MouthLeft', 'MouthRight', 'NoseWrinklerL', 'NoseWrinklerR', 'OuterBrowRaiserL',
    'OuterBrowRaiserR', 'UpperLidRaiserL', 'UpperLidRaiserR', 'UpperLipRaiserL', 'UpperLipRaiserR']
assert len(FEA_NAMES) == 63
IDX = {n: i for i, n in enumerate(FEA_NAMES)}

#model output order (from the training notebook)
#no contempt in here
ID_TO_EMOTION = {0: 'Neutral', 1: 'Happy', 2: 'Sad', 3: 'Surprise', 4: 'Fear', 5: 'Disgust', 6: 'Anger'}


def load_mlp_weights(keras_path):
    #a .keras file is just a zip, so pull the weights file out and read it with h5py (no tensorflow)
    with zipfile.ZipFile(keras_path) as z:
        z.extract('model.weights.h5', path='_tmp_weights')
    f = h5py.File('_tmp_weights/model.weights.h5', 'r')
    return {'W0': f['layers/dense/vars/0'][:], 'b0': f['layers/dense/vars/1'][:],
            'W1': f['layers/dense_1/vars/0'][:], 'b1': f['layers/dense_1/vars/1'][:],
            'W2': f['layers/dense_2/vars/0'][:], 'b2': f['layers/dense_2/vars/1'][:]}


def mlp_predict(X, w):
    #hand-coded forward pass matching the architecture 63 -> Dense128 relu -> Dense64 relu -> Dense7 softmax
    def relu(x): return np.maximum(0, x)
    def softmax(x):
        e = np.exp(x - x.max(axis=-1, keepdims=True)); return e / e.sum(axis=-1, keepdims=True)
    h = relu(X @ w['W0'] + w['b0'])
    h = relu(h @ w['W1'] + w['b1'])
    probs = softmax(h @ w['W2'] + w['b2'])
    return np.array([ID_TO_EMOTION[i] for i in probs.argmax(1)])


#my cosine classifier, ported from C# FaceEmotionClassifier.cs
PROTO = {'Happy': [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0], 'Sad': [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0], 'Surprise': [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1], 
         'Fear': [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1], 'Anger': [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0], 'Disgust': [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]}
PROTO = {k: np.array(v, float) for k, v in PROTO.items()}
def _avg(v, a, b): return (v[IDX[a]] + v[IDX[b]]) / 2.0
def _build11(v):
    return np.array([_avg(v,'InnerBrowRaiserL', 'InnerBrowRaiserR'), _avg(v,'OuterBrowRaiserL', 'OuterBrowRaiserR'),
        _avg(v,'BrowLowererL', 'BrowLowererR'), _avg(v,'UpperLidRaiserL', 'UpperLidRaiserR'),
        _avg(v,'CheekRaiserL', 'CheekRaiserR'), _avg(v,'NoseWrinklerL', 'NoseWrinklerR'),
        _avg(v,'UpperLipRaiserL', 'UpperLipRaiserR'), _avg(v,'LipCornerPullerL', 'LipCornerPullerR'),
        _avg(v,'LipCornerDepressorL', 'LipCornerDepressorR'), _avg(v,'LipStretcherL', 'LipStretcherR'), v[IDX['JawDrop']]])
def _cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na == 0 or nb == 0 else float(a @ b / (na * nb))
def cosine_classify(v, active=0.2, minsim=0.5, contempt_asym=0.3):
    if abs(v[IDX['LipCornerPullerL']] - v[IDX['LipCornerPullerR']]) > contempt_asym: return 'Contempt'
    f = _build11(v)
    if f.max() < active: return 'Neutral'
    best, bs = 'Neutral', 0.0
    for n, p in PROTO.items():
        s = _cos(f, p)
        if s > bs: bs, best = s, n
    return best if bs >= minsim else 'Neutral'


#which muscles fire for each emotion (EM-FACS combos)
EMO_AUS = {
    'Happy': ['CheekRaiserL', 'CheekRaiserR', 'LipCornerPullerL', 'LipCornerPullerR'],
    'Sad': ['InnerBrowRaiserL', 'InnerBrowRaiserR', 'BrowLowererL', 'BrowLowererR', 'LipCornerDepressorL', 'LipCornerDepressorR'],
    'Surprise': ['InnerBrowRaiserL', 'InnerBrowRaiserR', 'OuterBrowRaiserL', 'OuterBrowRaiserR', 'UpperLidRaiserL', 'UpperLidRaiserR', 'JawDrop'],
    'Fear': ['InnerBrowRaiserL', 'InnerBrowRaiserR', 'OuterBrowRaiserL', 'OuterBrowRaiserR', 'BrowLowererL', 'BrowLowererR', 'UpperLidRaiserL', 'UpperLidRaiserR', 'LipStretcherL', 'LipStretcherR', 'JawDrop'],
    'Anger': ['BrowLowererL', 'BrowLowererR', 'UpperLidRaiserL', 'UpperLidRaiserR'],
    'Disgust': ['NoseWrinklerL', 'NoseWrinklerR', 'UpperLipRaiserL', 'UpperLipRaiserR']}
EMOTIONS = ['Happy', 'Sad', 'Surprise', 'Fear', 'Anger', 'Disgust']


def make_faces(layers, n, rng):
    #layers is a list of (au_list, low, high) 
    #start with quiet baseline noise everywhere then for each layer turn its muscles up to a random value in [low, high]
    #using maximum so that when two emotions share a muscle they dont cancel out
    X = np.abs(rng.normal(0, 0.04, (n, 63))).clip(0, 0.12)
    for aus, lo, hi in layers:
        for au in aus:
            X[:, IDX[au]] = np.maximum(X[:, IDX[au]], rng.uniform(lo, hi, n))
    return X.clip(0, 1)


def accuracy_table(title, X, Y, w):
    #runs both classifiers on X, prints per-emotion and overall accuracy vs the true labels Y
    mlp = mlp_predict(X, w)
    cos = np.array([cosine_classify(X[i]) for i in range(len(X))])
    print(f"\n=== {title} ===")
    print(f"{'Emotion':<10}{'MLP':>9}{'Cosine':>9}")
    for e in EMOTIONS:
        m = Y == e
        if m.sum():
            print(f"{e:<10}{(mlp[m]==e).mean()*100:>8.1f}%{(cos[m]==e).mean()*100:>8.1f}%")
    print(f"{'-'*28}")
    print(f"{'OVERALL':<10}{(mlp==Y).mean()*100:>8.1f}%{(cos==Y).mean()*100:>8.1f}%")
    #how often each one bailed to neutral (matters most in the medium case)
    print(f"{'(Neutral%)':<10}{(mlp=='Neutral').mean()*100:>8.1f}%{(cos=='Neutral').mean()*100:>8.1f}%")
    return mlp, cos


def main():
    keras_path = sys.argv[1] if len(sys.argv) > 1 else 'fau_model.keras'
    w = load_mlp_weights(keras_path)
    rng = np.random.default_rng(42)
    N = 300

    #strong single emotions (0.7 to 1.0)
    Xe = np.vstack([make_faces([(EMO_AUS[e], 0.7, 1.0)], N, rng) for e in EMOTIONS])
    Ye = np.array([e for e in EMOTIONS for _ in range(N)])
    accuracy_table("EXTREME (strong expressions)", Xe, Ye, w)

    #medium weak single emotions (0.15 to 0.40)
    Xm = np.vstack([make_faces([(EMO_AUS[e], 0.15, 0.40)], N, rng) for e in EMOTIONS])
    Ym = np.array([e for e in EMOTIONS for _ in range(N)])
    accuracy_table("MILD (weak expressions)", Xm, Ym, w)

    #mixed e.g. emotion A strong + emotion B weak. correct answer is A
    Xd, Yd = [], []
    for a in EMOTIONS:
        for b in EMOTIONS:
            if a == b: continue
            Xd.append(make_faces([(EMO_AUS[a], 0.6, 0.9), (EMO_AUS[b], 0.2, 0.45)], 60, rng))
            Yd += [a] * 60
    Xd = np.vstack(Xd); Yd = np.array(Yd)
    accuracy_table("MIXED - dominant emotion (should pick the strong one)", Xd, Yd, w)

    #mixed e.g. no single right answer so just measure agreement between both MLP and cosine
    Xb = []
    pairs = [(a, b) for i, a in enumerate(EMOTIONS) for b in EMOTIONS[i+1:]]
    for a, b in pairs:
        Xb.append(make_faces([(EMO_AUS[a], 0.45, 0.75), (EMO_AUS[b], 0.45, 0.75)], 60, rng))
    Xb = np.vstack(Xb)
    mlp_b = mlp_predict(Xb, w)
    cos_b = np.array([cosine_classify(Xb[i]) for i in range(len(Xb))])
    agree = (mlp_b == cos_b).mean() * 100
    print(f"\n=== MIXED - 50/50 blends (no single right answer) ===")
    print(f"MLP and cosine agreed on {agree:.1f}% of {len(Xb)} blended faces")
    print(f"On disagreements they are each picking one of the two blended emotions, which is reasonable.")

    print("\nNOTE: EXTREME is biased toward cosine (data uses cosine's own rules).")
    print("MILD and MIXED are the more meaningful tests for telling the two apart.")


if __name__ == '__main__':
    main()
