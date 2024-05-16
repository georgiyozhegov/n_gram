# Info

Simple tool for training n-gram language model. Inspired by [this course](https://lena-voita.github.io/nlp_course.html).

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

# Examples

I've trained a trigram model on 20000 samples from the [Tiny Stories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset. Here are some examples of generated text:
- "\_\_sos_\_ Once upon a time a mom, a dad, a big sister, and a little girl below shouted, "Look Mama! A talking cloud!" The little girl opened her hand, and the monkey happily ate it all in one day. She was so kind he said yes and showed him the pin. "I poked you with this. It is a storm. The waves were so tall and wide, it seemed like something was calling her to come to an end eventually. They all had an incredible songbird inside. Billy was happy and excited. \_\_eos_\_"
- "\_\_sos_\_ Once upon a time there was a light girl with a basket. She then sent the basket to the washing machine. While the laundry was all hung up, Daisy and her family were getting ready to fly it, it suddenly flew away! The lion felt bad for being rude. He said, "It's my p
leasure. It's important to remember to forgive. \_\_eos_\_"
- "\_\_sos_\_ Once upon a time a family lived in a stream with many stones on the ground, it glistened in the sunshine. From that day forth they were always with her and learn with her and waved goodbye to Mommy. The bus driver was happy and flew away happily. Timmy felt proud of their pictures. \_\_eos_\_"

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

- [GitHub](https://github.com/georgiyozhegov/n_gram)
- [Crate](https://crates.io/crates/n_gram)
