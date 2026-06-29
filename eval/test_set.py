"""
Evaluation test set — 10 questions with ground-truth answers.

Ground truths are used by RAGAS ContextRecall to check whether the
retrieved context contains all the information needed to answer.
They are NOT shown to the LLM during generation — they are only used
as reference answers during offline evaluation.

Documents in the index:
  - attention1.pdf  : "Attention Is All You Need" (Vaswani et al., 2017)
  - sequence1.pdf   : Sequence-to-Sequence with Attention (Bahdanau et al.)
  - PYTHON1.pdf     : Python programming tutorial
  - ll1.pdf         : Large Language Models: A Survey
"""

TEST_QUESTIONS = [
    {
        "question": "What is the attention mechanism in neural networks?",
        "ground_truth": (
            "The attention mechanism maps a query and a set of key-value pairs to an output. "
            "The output is computed as a weighted sum of the values, where the weight assigned "
            "to each value is determined by a compatibility function of the query with the "
            "corresponding key."
        ),
    },
    {
        "question": "How does self-attention work in the Transformer?",
        "ground_truth": (
            "Self-attention, also called intra-attention, allows each position in a sequence "
            "to attend to all positions in the same sequence. The Transformer computes queries, "
            "keys, and values from the same input using learned linear projections, then applies "
            "scaled dot-product attention: softmax(QK^T / sqrt(d_k)) * V."
        ),
    },
    {
        "question": "What are the advantages of self-attention over recurrent layers?",
        "ground_truth": (
            "Self-attention layers have lower total computational complexity per layer than "
            "recurrent layers when the sequence length is smaller than the representation "
            "dimensionality. They also allow more computation to be parallelized since they "
            "require a constant number of sequential operations, whereas recurrent layers "
            "require O(n) sequential operations."
        ),
    },
    {
        "question": "What is multi-head attention and why is it used?",
        "ground_truth": (
            "Multi-head attention projects queries, keys, and values h times with different "
            "learned linear projections, runs attention in parallel on each projection, "
            "and concatenates the results. This allows the model to jointly attend to "
            "information from different representation subspaces at different positions."
        ),
    },
    {
        "question": "What is positional encoding and why does the Transformer need it?",
        "ground_truth": (
            "Since the Transformer contains no recurrence or convolution, positional encodings "
            "are added to the input embeddings to inject information about the position of tokens "
            "in the sequence. The paper uses sine and cosine functions of different frequencies: "
            "PE(pos, 2i) = sin(pos/10000^(2i/d_model)) and PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))."
        ),
    },
    {
        "question": "What is the difference between encoder and decoder in the Transformer?",
        "ground_truth": (
            "The encoder maps an input sequence to a sequence of continuous representations. "
            "The decoder generates an output sequence one element at a time using the encoder "
            "output. The decoder has an additional multi-head attention layer over the encoder "
            "output, and uses masking in its self-attention to prevent positions from attending "
            "to subsequent positions."
        ),
    },
    {
        "question": "What are sequence-to-sequence models used for?",
        "ground_truth": (
            "Sequence-to-sequence models are used for tasks where both input and output are "
            "variable-length sequences, such as machine translation, text summarization, "
            "speech recognition, and question answering. They typically use an encoder to "
            "compress the input into a fixed-size context vector and a decoder to generate output."
        ),
    },
    {
        "question": "How do you define a function in Python?",
        "ground_truth": (
            "In Python, a function is defined using the 'def' keyword followed by the function "
            "name, parentheses containing optional parameters, and a colon. The function body "
            "is indented. Example: def function_name(parameters): body. The function can "
            "optionally return a value using the 'return' statement."
        ),
    },
    {
        "question": "What are Python lists and how do you create one?",
        "ground_truth": (
            "A Python list is an ordered, mutable collection that can hold items of different "
            "data types. Lists are created using square brackets with comma-separated elements: "
            "my_list = [1, 'hello', 3.14]. Lists support indexing, slicing, and methods "
            "like append(), remove(), and sort()."
        ),
    },
    {
        "question": "What are large language models and how are they trained?",
        "ground_truth": (
            "Large language models (LLMs) are deep learning models with billions of parameters "
            "trained on massive text corpora using self-supervised learning. They are primarily "
            "based on the Transformer architecture and trained to predict the next token. "
            "Training involves pre-training on large datasets followed by fine-tuning or "
            "reinforcement learning from human feedback (RLHF) to align with human preferences."
        ),
    },
]
