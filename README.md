# Info
Simple tool for training n-gram language model. Inspired by [this course](https://lena-voita.github.io/nlp_course.html)

# Usage
```rust
use n_gram::*;

// Initializing model
let config = Config::default();
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

// Reset model
model.reset();

// Load model back
model.load("model.json").unwrap();
```

# Installation
```bash
cargo add n_gram
```
If you want to save & load your models:
```bash
cargo add n_gram --features=saveload
```
If you want to load tiny corpus for training:
```bash
cargo add n_gram --features=corpus
```

# Links
[github](https://github.com/georgiyozhegov/n_gram) <br>
[crates.io](https://crates.io/crates/n_gram)
