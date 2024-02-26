# Info
Simple tool for training n-gram language model. Inspired by [this course](https://lena-voita.github.io/nlp_course.html)

# Usage
```rust
use n_gram::*;

fn main() {
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
}
```
You can load a 4-gram model trained on 30000 samples from the [Tiny Stories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset:
```rust
use n_gram::*;

fn main() {
    let config = Config::new(4, DEFAULT_SMOOTHING, DEFAULT_SAMPLING);
    let mut model = Modell::new(config);

    model.load("models/tiny-stories-4g.json").unwrap();

    let mut tokens = sos(tokenize("Once upon a"));
    model.generate(&mut tokens, 100);

    println!("{}", tokens.join(" "));
}
```
Here are some examples of generated text:
- "\_\_sos_\_ Once upon a time two friends, John and Mary. Mary was very impatient and kept asking her mom and dad more. \_\_eos_\_"
- "\_\_sos_\_ Once upon a time in a small house, there lived a smart mouse. He loved to explore his house. One day, Tom met a little girl in tears. She was trying to soak the paper. But her plan didn't work and the paper started to rip and tear. She was so upset. She thought and thought about how brave she had been. Suddenly, she heard a funny sound in his chest. He feels something in his chest. He felt dizzy. He blacked out. Lily did not notice. She walked home with her mom. \_\_eos_\_"

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
