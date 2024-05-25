use rand::seq::SliceRandom;
use std::{
    cmp::max,
    collections::HashMap,
};
#[cfg(feature = "saveload")]
use std::fs::File;
#[cfg(feature = "saveload")]
use std::io::Write;

/// Start-Of-Sentence token.
pub const SOS: &str = "__sos__";
/// End-Of-Sentence token.
pub const EOS: &str = "__eos__";

pub const DEFAULT_CONTEXT: usize = 2;
pub const DEFAULT_SMOOTHING: bool = true;
pub const DEFAULT_SAMPLING: f32 = 0.8;

#[cfg(feature = "saveload")]
const SPLIT_TOKEN: &str = "<[SP]>";

/// Splits text by whitespaces.
pub fn tokenize(text: String) -> Vec<String>
{
        text.split_whitespace().map(|t| t.to_string()).collect()
}

/// Adds SOS token at the start.
pub fn sos(tokens: Vec<String>) -> Vec<String>
{
        let mut tokens = tokens;
        tokens.insert(0, SOS.to_string());
        tokens
}

/// Adds EOS token at the end.
pub fn eos(tokens: Vec<String>) -> Vec<String>
{
        let mut tokens = tokens;
        tokens.push(EOS.to_string());
        tokens
}

/// Creates list of n-grams from `tokens`.
///
/// # Usage
///
/// ```
/// use n_gram::{
///         n_grams,
///         tokenize,
/// };
///
/// let tokens = tokenize("Eat tasty cakes".to_string());
/// let n = 2; // bigrams (context window - one word)
/// let n_grams = n_grams(tokens, n);
///
/// assert_eq!(
///         n_grams,
///         vec![
///                 // first item - previous tokens, second - next token
///                 (vec!["Eat".to_string()], "tasty".to_string()),
///                 (vec!["tasty".to_string()], "cakes".to_string())
///         ]
/// );
/// ```
/// 
/// # Note
///
/// Assumes that number of `tokens` is greater than `n`.
pub fn n_grams(tokens: Vec<String>, n: usize) -> Vec<(Vec<String>, String)>
{
        let mut n_grams = Vec::new();
        for i in 0..tokens.len() - n + 1 {
                n_grams.push((
                        (&tokens[i..i + n - 1]).to_vec(),
                        (&tokens[i + n - 1]).to_owned(),
                ));
        }
        n_grams
}

/// Returns tiny corpus with very simple & short sentences.
///
/// # Usage
///
/// ```
/// use n_gram::tiny_corpus;
///
/// let corpus = tiny_corpus();
/// println!("First sentence: {}", corpus[0]);
/// ```
/// 
/// # Note
///
/// Size of corpus is about 50 samples. Also, this corpus is AI-generated.
#[cfg(feature = "corpus")]
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
/// # Usage
///
/// ```
/// use n_gram::Config;
///
/// // context size (n-1) to use
/// let context = 1;
///
/// // Smoothing (using backoff):
/// // - If you can, use trigrams
/// // - If not, use bigrams
/// // - If even bigrams does not help, use unigrams
/// let smoothing = true;
///
/// // top-p% of samples (in this case - 20%)
/// let sampling = 0.2;
///
/// let config = Config::new(context, smoothing, sampling);
/// ```
/// You can also use defaults:
/// ```
/// use n_gram::Config;
///
/// let config = Config::default();
/// ```
/// Or set specific default value if needed:
/// ```
/// use n_gram::{
///         Config,
///         DEFAULT_SAMPLING,
/// };
///
/// let config = Config::new(3, true, DEFAULT_SAMPLING);
/// ```
#[derive(Debug, Clone)]
pub struct Config
{
        context: usize,
        smoothing: bool,
        sampling: f32,
}

impl Config
{
        pub fn new(context: usize, smoothing: bool, sampling: f32) -> Self
        {
                Self {
                        context,
                        smoothing,
                        sampling,
                }
        }
}

impl Default for Config
{
        fn default() -> Self
        {
                Self {
                        context: 2,
                        smoothing: true,
                        sampling: 0.8,
                }
        }
}

/// N-gram language model.
///
/// # Usage
///
/// ```
/// use n_gram::*;
///
/// // Initializing model
/// let config = Config::default();
/// let mut model = Model::new(config);
///
/// // Loading and tokenizing corpus
/// let corpus = tiny_corpus()
///         .iter()
///         .map(|t| sos(eos(tokenize(t.to_owned()))))
///         .collect::<Vec<_>>();
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
/// // Reset model
/// model.reset();
///
/// // Load model back
/// model.load("model.json").unwrap();
/// ```
#[derive(Debug, Clone)]
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
                        for n_gram in n_grams(tokens, self.config.context + 1) {
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

impl Model
{
        /// Predicts next token for given `tokens`.
        ///
        /// # Usage
        ///
        /// ```
        /// use n_gram::{
        ///         tokenize,
        ///         Config,
        ///         Model,
        /// };
        ///
        /// let model = Model::new(Config::default()); // assuming that your model is trained.
        ///
        /// let tokens = tokenize("The quick brown".to_string());
        /// let next_token = model.predict(tokens);
        ///
        /// println!("{next_token}");
        /// ```
        pub fn predict(&self, tokens: Vec<String>) -> String
        {
                let tokens = cut(tokens, self.config.context);
                if let Some(counts) = self.get(tokens) {
                        {
                                let mut counts = counts.iter().collect::<Vec<_>>();
                                counts.sort_by(|a, b| b.1.cmp(&a.1));
                                let samples = max(
                                        1,
                                        (counts.len() as f32 * self.config.sampling) as usize,
                                ); // at least one sample
                                counts.into_iter()
                                        .map(|(k, _)| k)
                                        .take(samples)
                                        .collect::<Vec<_>>()
                        }
                        .choose(&mut rand::thread_rng())
                        .unwrap()
                }
                else {
                        EOS
                }
                .to_string()
        }

        /// Generates tokens using [`Model::predict()`].
        ///
        /// # Usage
        ///
        /// ```
        /// use n_gram::{
        ///         tokenize,
        ///         Config,
        ///         Model,
        /// };
        ///
        /// let model = Model::new(Config::default()); // assuming that your model is trained.
        /// let mut tokens = tokenize("The quick brown".to_string());
        /// let max = 10; // max 10 generated tokens.
        ///
        /// model.generate(&mut tokens, max);
        ///
        /// println!("{tokens:?}");
        /// ```
        ///
        /// # Note
        ///
        /// Stops if model predicts the EOS token.
        pub fn generate(&self, tokens: &mut Vec<String>, max: u32)
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

#[cfg(feature = "saveload")]
impl Model
{
        /// Saves model into json file.
        /// 
        /// # Note
        ///
        /// Returns file.write() status code.
        fn save(&self, path: &str) -> Result_<usize>
        {
                let mut file = File::create(path)?;
                let model = self
                        .model
                        .iter()
                        .map(|(k, v)| (k.join(SPLIT_TOKEN), v))
                        .collect::<Vec<_>>();
                let model = serde_json::to_string(&model)?;
                file.write(model.as_bytes())?;
                Ok(0)
        }

        /// Loads model from json file.
        fn load(&mut self, path: &str) -> Result_<()>
        {
                let file = File::open(path)?;
                let model: Vec<(String, HashMap<String, u32>)> = serde_json::from_reader(file)?;
                self.model = model
                        .iter()
                        .map(|(k, v)| {
                                (
                                        k.split(SPLIT_TOKEN)
                                                .map(|t| t.to_string())
                                                .collect::<Vec<_>>(),
                                        v.to_owned(),
                                )
                        })
                        .collect::<HashMap<Vec<String>, HashMap<String, u32>>>();
                Ok(())
        }
}
