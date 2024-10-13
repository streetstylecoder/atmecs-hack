from transformers import pipeline
import torch

device = 0 
#if torch.cuda.is_available() else -1  # Use GPU if available, else CPU
print(device)
fill_mask = pipeline(
    "fill-mask",
    model='DeepChem/ChemBERTa-77M-MLM',
    tokenizer='mrm8488/chEMBL_smiles_v1',
    device=device
)

# CC(C)CN(CC(OP(=O)(O)O)C(Cc1ccccc1)NC(=O)OC1CCOC1)S(=O)(=O)c1ccc(N)cc1 Atazanavir
smile1 = "CC(=O)N1c2ccc(-c3ccc(C(=O)O)cc3)cc2[C@H](Nc2ccc(Cl)cc2)C[C@@H]1C<mask>"

abc =fill_mask(smile1)
print(abc)

