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
You can load a 4-gram model trained on 20000 samples from the [Tiny Stories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset:
```rust
use n_gram::*;

fn main() {
    let config = Config::new(3, DEFAULT_SMOOTHING, DEFAULT_SAMPLING);
    let mut model = Modell::new(config);

    model.load("models/tiny-stories-4n.json").unwrap();

    let mut tokens = sos(tokenize("Once upon a"));
    model.generate(&mut tokens, 100);

    println!("{}", tokens.join(" "));
}
```
Here are some examples of generated text:
- "\_\_sos_\_ Once upon a time two friends, John and Bob. John was sad and she did a great job. Thank you for lending me Mittens." Lily smiled and forgot about being scary, and he handed Lucy a pencil. Lucy was happy, and felt that everything was right in the spot so she wouldn't make a mess. Can we fix the puzzle together?" Lily stopped crying. She realized that her dream was not just soaring, but experiencing life from both the sky and Jack felt proud that she had such a wonderful adventure and it was because he hadn't exercised. She told him he should think"
- "\_\_sos_\_ Once upon a time in a big book with many pages. He learned about animals, colors, and shapes. He liked this nap so much that she ate it with Lily. They were scared and angry by the noise of a thing in the sky too. Its light was warm and inviting. She planted her feet and gave her some celery for a snack. \_\_eos_\_"

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
