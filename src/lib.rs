use rand::seq::SliceRandom;
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::Write;



type Result_<T> = Result<T, Box<dyn Error>>;


/// Start-Of-Sentence token.
pub const SOS: &str = "__sos__";


/// End-Of-Sentence token.
pub const EOS: &str = "__eos__";


/// Splits text by whitespaces
pub fn tokenize(text: String) -> Vec<String>
{
      text.split_whitespace().map(|t| t.to_string()).collect()
}


/// Adds SOS token at the start
pub fn sos(tokens: Vec<String>) -> Vec<String>
{
      let mut tokens = tokens;
      tokens.insert(0, SOS.to_string());
      tokens
}


/// Adds EOS token at the end
pub fn eos(tokens: Vec<String>) -> Vec<String>
{
      let mut tokens = tokens;
      tokens.push(EOS.to_string());
      tokens
}


/// Creates list of n-grams from tokens
///
/// ```rust
/// use n_gram::{
///       n_grams,
///       tokenize,
/// };
///
/// let tokens = tokenize("Eat tasty cakes".to_string());
/// let n = 2; // bigrams (context window - one word)
/// let n_grams = n_grams(tokens, n);
///
/// assert_eq!(
///       n_grams,
///       vec![
///             // first item - previous tokens, second - next token
///             (vec!["Eat".to_string()], "tasty".to_string()),
///             (vec!["tasty".to_string()], "cakes".to_string())
///       ]
/// );
/// ```
///
/// Assumes that number of tokens is greater than n.
pub fn n_grams(tokens: Vec<String>, n: usize) -> Vec<(Vec<String>, String)>
{
      let mut n_grams = Vec::new();
      for i in 0..tokens.len() - n + 1 {
            n_grams.push(((&tokens[i..i + n - 1]).to_vec(), (&tokens[i + n - 1]).to_owned()));
      }
      n_grams
}


/// Returns tiny corpus with very simple & short sentences.
///
/// ```rust
/// use n_gram::tiny_corpus;
///
/// let corpus = tiny_corpus();
/// println!("First sentence: {}", corpus[0]);
/// ```
///
/// Size of corpus is about 50 samples. Also, this corpus is AI-generated.
pub fn tiny_corpus() -> Vec<String>
{
      vec![
            "The quick brown fox jumps over the lazy dog.",
            "A speedy red fox leaps above the tired hound.",
            "Swift brown foxes hop across the sleepy canine.",
            "A fast orange fox vaults over the dozing pooch.",
            "The nimble tan fox jumps past the resting dog.",
            "An agile russet fox springs over the snoozing pup.",
            "Quick brown foxes leap over lazy dogs.",
            "A rapid golden fox vaults above the slumbering pet.",
            "The hasty copper-colored fox jumps over the dozing animal.",
            "A brisk chestnut fox leaps past the resting puppy.",
            "The sly brown fox jumps over the lazy hound.",
            "A swift red fox leaps above the tired dog.",
            "Nimble gray foxes hop across the sleepy pup.",
            "A quick silver fox vaults over the dozing canine.",
            "The agile black fox jumps past the resting hound.",
            "An elegant white fox springs over the snoozing pet.",
            "Speedy brown foxes leap over lazy cats.",
            "A nimble golden fox vaults above the slumbering feline.",
            "The graceful silver fox jumps over the dozing kitten.",
            "A sleek black fox leaps past the resting tabby.",
            "The clever brown fox jumps over the drowsy dog.",
            "A lively red fox leaps above the weary hound.",
            "Brisk brown foxes hop across the sluggish canines.",
            "A spry orange fox vaults over the sleeping pooches.",
            "The energetic tan fox jumps past the lounging pups.",
            "An active russet fox springs over the dozing pets.",
            "Quick brown foxes leap over lazy felines.",
            "A rapid golden fox vaults above the slumbering cats.",
            "The vigorous copper-colored fox jumps over the dozing animals.",
            "A spirited chestnut fox leaps past the resting creatures.",
            "The cunning brown fox jumps over the lax hounds.",
            "A fleet red fox leaps above the fatigued dogs.",
            "Agile gray foxes hop across the sluggish puppies.",
            "A deft silver fox vaults over the sleeping canids.",
            "The nimble black fox jumps past the relaxed hounds.",
            "An alert white fox springs over the dozing felines.",
            "Speedy brown foxes leap over lazy primates.",
            "A skillful golden fox vaults above the slumbering apes.",
            "The adept silver fox jumps over the dozing mammals.",
            "A savvy black fox leaps past the resting marsupials.",
            "The smart brown fox jumps over the lethargic dog.",
            "A swift crimson fox vaults above the weary hound.",
            "Agile brownish-red foxes hop across the drowsy canine.",
            "A quick tawny fox springs over the snoozing pooch.",
            "The nimble chocolate-colored fox jumps past the resting doggy.",
            "An alert sandy-colored fox leaps above the slumbering pet.",
      ]
      .iter()
      .map(|t| t.to_string())
      .collect::<Vec<_>>()
}


fn cut(tokens: Vec<String>, context: usize) -> Vec<String>
{
      if tokens.len() <= context {
            tokens
      }
      else {
            tokens[tokens.len() - context..].to_vec()
      }
}


/// Config for Model.
///
/// ```rust
/// use n_gram::Config;
///
/// let context = 1; // bigrams (context = n - 1)
/// let smoothing = true;
/// // Smoothing (using backoff):
/// // - If you can, use trigrams
/// // - If not, use bigrams
/// // - If even bigrams does not help, use unigrams
/// let config = Config::new(context, smoothing);
/// ```
pub struct Config
{
      context: usize,
      smoothing: bool,
}

impl Config
{
      pub fn new(context: usize, smoothing: bool) -> Self
      {
            Self {
                  context,
                  smoothing,
            }
      }
}


/// N-gram language model.
///
/// ```rust
/// use n_gram::*;
///
/// // Initializing model
/// let config = Config::new(3, true);
/// let mut model = Model::new(config);
///
/// // Loading and tokenizing corpus
/// let corpus = tiny_corpus()
///       .iter()
///       .map(|t| sos(eos(tokenize(t.to_owned()))))
///       .collect::<Vec<_>>();
///
/// model.train(corpus);
///
/// // Now you are ready to generate something
/// let mut tokens = sos(tokenize("The quick".to_string()));
/// let max = 10; // max number of generated tokens
/// model.generate(&mut tokens, max);
///
/// // Save model
/// model.save("model.json").unwrap();
///
/// // Load model back
/// model.load("model.json").unwrap();
/// ```
pub struct Model
{
      config: Config,
      model: HashMap<Vec<String>, HashMap<String, u32>>,
}

impl Model
{
      pub fn new(config: Config) -> Self
      {
            Self {
                  config,
                  model: HashMap::new(),
            }
      }

      fn get(&self, tokens: Vec<String>) -> Option<&HashMap<String, u32>>
      {
            if let Some(tokens_) = self.model.get(&tokens) {
                  Some(tokens_)
            }
            else if self.config.smoothing && tokens.len() > 0 {
                  // Backoff
                  self.get(cut(tokens.clone(), tokens.len() - 1))
            }
            else {
                  None
            }
      }

      /// Trains model.
      pub fn train(&mut self, corpus: Vec<Vec<String>>)
      {
            for tokens in corpus {
                  let tokens = sos(eos(tokens));
                  for n_gram in n_grams(tokens, self.config.context - 1) {
                        if let Some(counts) = self.model.get_mut(&n_gram.0) {
                              if let Some(count) = counts.get_mut(&n_gram.1) {
                                    *count += 1;
                              }
                              else {
                                    counts.insert(n_gram.1, 1);
                              }
                        }
                        else {
                              self.model.insert(n_gram.0.clone(), HashMap::new());
                              self.model.get_mut(&n_gram.0).unwrap().insert(n_gram.1, 1);
                        }
                  }
            }
      }

      /// Resets model parameters.
      pub fn reset(&mut self)
      {
            self.model = HashMap::new();
      }
}

impl SaveLoad for Model
{
      /// Saves model into json file.
      ///
      /// Returns file.write() status code.
      fn save(&self, path: &str) -> Result_<usize>
      {
            let mut file = File::create(path)?;
            let keys = self.model.keys().map(|k| k.join(" "));
            let values = self.model.values();
            let model = keys.zip(values).collect::<Vec<_>>();
            let model = serde_json::to_string(&model)?;
            file.write(model.as_bytes())?;
            Ok(0)
      }

      /// Loads model from json file.
      fn load(&mut self, path: &str) -> Result_<()>
      {
            let file = File::open(path)?;
            let model: Vec<(String, HashMap<String, u32>)> = serde_json::from_reader(file)?;
            let mut model_ = HashMap::new();
            for (key, value) in model.iter() {
                  model_.insert(
                        key.split_whitespace().map(|t| t.to_string()).collect::<Vec<_>>(),
                        value.clone(),
                  );
            }
            self.model = model_;
            Ok(())
      }
}

impl Predict for Model
{
      /// Predicts next token for given tokens.
      ///
      /// ```
      /// use n_gram::{
      ///       tokenize,
      ///       Config,
      ///       Model,
      ///       Predict, // trait for predict()
      /// };
      ///
      /// let model = Model::new(Config::new(3, true)); // assuming that your model is trained.
      ///
      /// let tokens = tokenize("The quick brown".to_string());
      /// let next_token = model.predict(tokens);
      ///
      /// println!("{next_token}");
      /// ```
      fn predict(&self, tokens: Vec<String>) -> String
      {
            let tokens = cut(tokens, self.config.context);
            if let Some(counts) = self.get(tokens) {
                  {
                        let mut counts = counts.iter().collect::<Vec<_>>();
                        counts.sort_by(|a, b| b.1.cmp(&a.1));
                        counts.into_iter().map(|(k, _)| k).collect::<Vec<_>>()
                  }
                  .choose(&mut rand::thread_rng())
                  .unwrap()
            }
            else {
                  EOS
            }
            .to_string()
      }
}

impl Generate for Model
{
      /// Generates tokens using Model::predict().
      ///
      /// ```
      /// use n_gram::{
      ///       tokenize,
      ///       Config,
      ///       Generate, // trait for generate()
      ///       Model,
      /// };
      ///
      /// let model = Model::new(Config::new(3, true)); // assuming that your model is trained.
      /// let mut tokens = tokenize("The quick brown".to_string());
      /// let max = 10; // max 10 generated tokens.
      ///
      /// model.generate(&mut tokens, max);
      ///
      /// println!("{tokens:?}");
      /// ```
      ///
      /// Stops if model predicts the EOS token.
      fn generate(&self, tokens: &mut Vec<String>, max: u32)
      {
            for _ in 0..max {
                  let token = self.predict(cut(tokens.to_vec(), self.config.context));
                  tokens.push(token.clone());
                  if token == EOS {
                        break;
                  }
            }
      }
}


pub trait Predict
{
      fn predict(&self, tokens: Vec<String>) -> String;
}


pub trait Generate
{
      fn generate(&self, tokens: &mut Vec<String>, max: u32);
}


pub trait SaveLoad
{
      fn save(&self, path: &str) -> Result_<usize>;
      fn load(&mut self, path: &str) -> Result_<()>;
}
