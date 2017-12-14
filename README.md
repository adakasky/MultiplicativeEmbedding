# SemanticWordEmbedding
Try to solve the problem of embeddings on negation and degree modifier.
Traditional word embeddings are trained unsupervisedly on large corpus, so they only capture the syntactic co-occurrence information and lack of semantic interpretation. Supporting additive rules of vectors, they can hardly solve the problem of negation and degree modifiers.
My idea is try to supervisedly train word embeddings on the Stanford Natural Language Inference (SNLI) Corpus, so that either entailment or contradiction relations can provide some meaning to word representations.
