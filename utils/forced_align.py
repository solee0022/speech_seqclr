import torch
import torchaudio
from dataclasses import dataclass
import matplotlib.pyplot as plt
import json
import IPython
import re

import syllables as syll_esti
import spacy
from spacy_syllables import SpacySyllables
import nltk
from nltk.corpus import cmudict
from nltk.tokenize import word_tokenize
from num2words import num2words
from unidecode import unidecode
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model().to(device)
labels = bundle.get_labels()
print(labels)

# 0. Prepare
## 0.0. load dataset
def load_data_from_json(jsonl_path):
    with open(jsonl_path, "r") as f:
        jsonl_data = list(f)

    data = []
    for pair in jsonl_data:
        sample = json.loads(pair)
        data.append(sample)
    return data

# jsonl_path = "/home/solee0022/data/UASpeech/doc/UASpeech_scripts_train_UW.jsonl"
# align_f = open("/home/solee0022/seqclr_exp/speech_seqclr/seqclr/dataset/UASpeech/UASpeech_scripts_train_UW_alignment.jsonl", "w")

data_path = "/home/solee0022/seqclr_exp/speech_seqclr/seqclr/dataset/accent_splits"
jsonl_path = data_path + "/original/train_random_100h.jsonl"
align_f = open(f"{data_path}/segment/train_random_100h.jsonl", "w")
data = load_data_from_json(jsonl_path)


def preprocess_word(word):
    preprocessed_word = word.replace("-", " ")
    preprocessed_word = preprocessed_word.replace(" & ", " ")
    preprocessed_word = preprocessed_word.replace(" !", "")
    preprocessed_word = preprocessed_word.replace(" ?", "")
    preprocessed_word = preprocessed_word.replace(" .", "")
    preprocessed_word = preprocessed_word.replace(". .", "")
    preprocessed_word = preprocessed_word.replace("--", " ")
    preprocessed_word = preprocessed_word.replace(" - ", " ")
    preprocessed_word = re.sub('[-_.=+;,→#/\?:^.&%@*\"※~ㆍ!』‘|\(\)\[\]`…》\”\“\’\'·]', '', preprocessed_word)
    preprocessed_word= unidecode(preprocessed_word)
    return preprocessed_word


def main(transcript, audio_path):
    # 0. load audio
    with torch.inference_mode():
        waveform, _ = torchaudio.load(audio_path)
        emissions, _ = model(waveform.to(device))
        emissions = torch.log_softmax(emissions, dim=-1)

    emission = emissions[0].cpu().detach()
    # 1. Generate alignment probability (trellis)
    # We enclose the transcript with space tokens, which represent SOS and EOS.
    dictionary = {c: i for i, c in enumerate(labels)}

    tokens = [dictionary[c] for c in transcript]

    def get_trellis(emission, tokens, blank_id=0):
        num_frame = emission.size(0)
        num_tokens = len(tokens)

        trellis = torch.zeros((num_frame, num_tokens))
        trellis[1:, 0] = torch.cumsum(emission[1:, blank_id], 0)
        trellis[0, 1:] = -float("inf")
        trellis[-num_tokens + 1 :, 0] = float("inf")

        for t in range(num_frame - 1):
            trellis[t + 1, 1:] = torch.maximum(
                # Score for staying at the same token
                trellis[t, 1:] + emission[t, blank_id],
                # Score for changing to the next token
                trellis[t, :-1] + emission[t, tokens[1:]],
            )
        return trellis


    trellis = get_trellis(emission, tokens)


    # 2. Find the most likely path (backtracking)
    @dataclass
    class Point:
        token_index: int
        time_index: int
        score: float


    def backtrack(trellis, emission, tokens, blank_id=0):
        t, j = trellis.size(0) - 1, trellis.size(1) - 1

        path = [Point(j, t, emission[t, blank_id].exp().item())]
        while j > 0:
            # Should not happen but just in case
            assert t > 0

            # 1. Figure out if the current position was stay or change
            # Frame-wise score of stay vs change
            p_stay = emission[t - 1, blank_id]
            p_change = emission[t - 1, tokens[j]]

            # Context-aware score for stay vs change
            stayed = trellis[t - 1, j] + p_stay
            changed = trellis[t - 1, j - 1] + p_change

            # Update position
            t -= 1
            if changed > stayed:
                j -= 1

            # Store the path with frame-wise probability.
            prob = (p_change if changed > stayed else p_stay).exp().item()
            path.append(Point(j, t, prob))

        # Now j == 0, which means, it reached the SoS.
        # Fill up the rest for the sake of visualization
        while t > 0:
            prob = emission[t - 1, blank_id].exp().item()
            path.append(Point(j, t - 1, prob))
            t -= 1

        return path[::-1]


    path = backtrack(trellis, emission, tokens)

    # 3. Segment the path
    # Merge the labels
    @dataclass
    class Segment:
        label: str
        start: int
        end: int
        score: float

        def __repr__(self):
            return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

        @property
        def length(self):
            return self.end - self.start
        
        @property
        def timestamp_return(self):
            return self.start, self.end


    def merge_repeats(path):
        i1, i2 = 0, 0
        segments = []
        while i1 < len(path):
            while i2 < len(path) and path[i1].token_index == path[i2].token_index:
                i2 += 1
            score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
            segments.append(
                Segment(
                    transcript[path[i1].token_index],
                    path[i1].time_index,
                    path[i2 - 1].time_index + 1,
                    score,
                )
            )
            i1 = i2
        return segments

    # What we want! 
    segments = merge_repeats(path)
    ctc_segments = []
    for seg in segments:
        ctc_segments.append([seg.timestamp_return[0], seg.timestamp_return[1]])
       
    # 4. Merge the segments into words
    # Merge words
    def merge_words(segments, separator="|"):
        words = []
        i1, i2 = 0, 0
        while i1 < len(segments):
            if i2 >= len(segments) or segments[i2].label == separator:
                if i1 != i2:
                    segs = segments[i1:i2]
                    word = "".join([seg.label for seg in segs])
                    score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                    words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
                i1 = i2 + 1
                i2 = i1
            else:
                i2 += 1
        return words


    # word_segments = merge_words(segments)
    
    return ctc_segments


# word to syllables
def segmentation():        
    # Initialize the dictionary
    d = cmudict.dict()
    nlp = spacy.load('en_core_web_sm')
    syllables = SpacySyllables(nlp)
    nlp.add_pipe('syllables', after='tagger')

    # Download necessary NLTK resources
    nltk.download('punkt')
    
    # word to syllable
    def split_syllables(word):
        try:
            # Get phonetic transcription(s) from CMU Dict
            phonemes = d[word.lower()][0]  # Taking the first pronunciation if multiple exist

            # Initialize syllable list and current syllable
            syllables = []
            current_syllable = []

            # Process each phoneme
            for phoneme in phonemes:
                current_syllable.append(phoneme)
                # Phonemes that indicate a new syllable start with a number (stress)
                if phoneme[-1].isdigit():
                    # Join the current syllable phonemes and start a new syllable
                    syllables.append(''.join(current_syllable))
                    current_syllable = []

            # Join all phonemes to form rough syllable-like segments
            syllable_words = '-'.join(syllables)
            return syllable_words
        except KeyError:
            return "Word not found in CMU Dictionary."


    # word to morpheme
    def split_morphemes(word):
        # Tokenize the sentence into words
        words = word_tokenize(word)

        # List of common English suffixes
        suffixes = {'ed', 'ing', 'ly', 'es', 's', 'er', 'ion', 'tion', 'ment', 'ness', 'ship', 'able', 'ible', 'al', 'ful', 'ish', 'ive', 'less', 'ous', 'ist', 'ism'}

        morphemes = []
        for word in words:
            # Find the longest possible suffix
            word_suffixes = [suffix for suffix in suffixes if word.endswith(suffix)]
            if word_suffixes:
                longest_suffix = max(word_suffixes, key=len)
                # Split the word into root and suffix
                root = word[:-len(longest_suffix)]
                morphemes.append(root)
                morphemes.append(longest_suffix)
            else:
                morphemes.append(word)
        return morphemes
    
    def spacy_syllablize(word):
        token = nlp(word)[0]
        return token._.syllables

    for line in data:
        audio_path = "/home/solee0022/data/cv/en/clips/"
        transcript = line['sentence']
        # transcript = transcript.replace("-", " ")
        # transcript = transcript.replace(" & ", " ")
        # transcript = transcript.replace(". .", "")
        # transcript = re.sub('[-_.=+;,#/\?:^.&%@*\"※~ㆍ!』‘|\(\)\[\]`…》\”\“\’\'·]', '', transcript)
        transcript = preprocess_word(transcript)
        transcript = "|" + transcript.upper() + "|"
        transcript = transcript.replace(" ", "|")
        
        ctc_segments = main(transcript, audio_path + line['path'])

        # syllable
        syll_seg = []
        sylls = []
        idx_s = 1
        splited_transcript = transcript.lower().split("|")[1:-1]
        splited_transcript = [wo for wo in splited_transcript if wo != ""]
        print(splited_transcript)
        if len(splited_transcript) > 1:
            for word in splited_transcript:
                syllable_words = spacy_syllablize(word)
                for syll in syllable_words:
                    idx_e = idx_s + len(syll)
                    seg = [ctc_segments[idx_s][0], ctc_segments[idx_e][1]]
                    syll_seg.append(seg)
                    sylls.append(syll)
                    idx_s = idx_e
                idx_s+=1
        else:
            word = splited_transcript[0]
            syllable_words = spacy_syllablize(word)
            for syll in syllable_words:
                idx_e = idx_s + len(syll)
                seg = [ctc_segments[idx_s][0], ctc_segments[idx_e][1]]
                syll_seg.append(seg)
                sylls.append(syll)
                idx_s = idx_e
            
        # morpheme
        morp_seg = []
        morps = []
        idx_s = 1
        if len(splited_transcript) > 1:
            for word in splited_transcript:
                morphemes = split_morphemes(word)
                for morp in morphemes:
                    idx_e = idx_s + len(morp)
                    seg = [ctc_segments[idx_s][0], ctc_segments[idx_e][1]]
                    morp_seg.append(seg)
                    morps.append(morp)
                    idx_s = idx_e
                idx_s+=1
        else:
            word = splited_transcript[0]
            morphemes = split_morphemes(word)
            for morp in morphemes:
                idx_e = idx_s + len(morp)
                seg = [ctc_segments[idx_s][0], ctc_segments[idx_e][1]]
                morp_seg.append(seg)
                morps.append(morp)
                idx_s = idx_e
                
        
        # new_line = {'audio': line['path'], 'text': line['sentence'], "label": line['label'], 'syllable': syll_seg, 'morpheme': morp_seg}
        new_line = {'audio': line['path'], 'text': line['sentence'], 'syllable': syll_seg, 'morpheme': morp_seg, 'syll': sylls, 'morp': morps}
        print(new_line)
        json.dump(new_line, align_f, ensure_ascii=False)
        align_f.write("\n")

segmentation()