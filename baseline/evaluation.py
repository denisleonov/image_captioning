import torch
from tqdm.auto import tqdm
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')

def get_predictions(encoder, decoder, dataset, idxs2sentence):
    predictions = []
    for img, _ in tqdm(dataset):
        img = img.to(device)
        feature = encoder(img.unsqueeze(0))
        predictions.append(idxs2sentence(decoder.sample(feature)))
    return predictions

def get_bleus(predictions, dataset, tokenize):
    score1 = 0
    score4 = 0
    smoother = SmoothingFunction()
    for candidate, (_, captions) in tqdm(zip(predictions, dataset), total=len(predictions)):
        candidate = tokenize(candidate)
        references = list(map(tokenize, captions))
        score1 += sentence_bleu(references, candidate, weights=(1, 0, 0, 0), smoothing_function=smoother.method1)
        score4 += sentence_bleu(references, candidate, weights=(0, 0, 0, 1), smoothing_function=smoother.method1)
    bleu1 = 100*score1/len(dataset)
    bleu4 = 100*score4/len(dataset)        
    print("BLEU 1:", np.round(bleu1,2), 
          "BLEU 4:", np.round(bleu4,2))    
    return bleu1, bleu4