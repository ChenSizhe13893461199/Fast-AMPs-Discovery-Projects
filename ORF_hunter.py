# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 11:44:48 2022

@author: Sizhe Chen
"""

import os
import string
pos_dict = {}
a=''
b=''
i=0
with open("your file.fasta") as infile:
    for line in infile.readlines():
        if line.startswith(">")==1:
         s=0
         key = line
         a=''
         print(i)
         i=i+1
        else:
             if "N" in line:
                 continue
             else:
              pos_dict[key] = a+line.strip("\n")
              a=pos_dict[key]
              

outfile = open('RNA-seq1.csv', 'w')

for key in list(pos_dict.keys())[:]:

    outfile.write(key)
    outfile.write(pos_dict[key]+"\n")

outfile.close()

#Searching ORF and Return Translated Peptides Sequences (11-50 Amino Acids Long)
codon_table = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
    'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
    'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
    'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
    'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
    'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
    'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
    'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'', 'TAG':'',
    'TGC':'C', 'TGT':'C', 'TGA':'', 'TGG':'W',
}

def translate_codon(codon):
    return codon_table.get(codon.upper(), '')

def find_orfs_with_trans(sequence, start_codons=['ATG'], stop_codons=['TAA', 'TAG', 'TGA']):
    orfs = []
    protein_seq = []
    start_index = -1
    for i in range(0,len(sequence)):
        codon = sequence[i:i+3]
        if codon in start_codons:
            start_index = i
        elif codon in stop_codons and (i+3-start_index)%3==0 and start_index != -1:
            if 33 <= (i+3-start_index-3) <= 150:  
                orf = sequence[start_index:i+3]
                protein = ''.join([translate_codon(orf[j:j+3]) for j in range(0, len(orf), 3)])
                orfs.append(orf)  
                protein_seq.append(protein)
            start_index = -1
    return orfs, protein_seq

peptide=[]
ORF=[]
for key in list(pos_dict.keys())[:]:

 orfs, protein_seq = find_orfs_with_trans(pos_dict[key])
 for i in range(0,len(protein_seq)):
  peptide.append(protein_seq[i])
  ORF.append(orfs[i])