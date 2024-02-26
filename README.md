# Info
Simple tool for training n-gram language model.

# Usage
```rust
use n_gram::*;

// Initializing model
let config = Config::new(3, true);
let mut model = Model::new(config);

// Loading and tokenizing corpus
let corpus = tiny_corpus()
    .iter()
    .map(|t| sos(eos(tokenize(t.to_owned()))))
    .collect::<Vec<_>>();
model.train(corpus);

// Now you are ready to generate something
let mut tokens = sos(tokenize("The quick".to_string()));
let max = 10; // max number of generated tokens
model.generate(&mut tokens, max);

// Save model
model.save("model.json").unwrap();

// Load model back
model.load("model.json").unwrap();
```
