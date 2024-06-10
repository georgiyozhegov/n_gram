use n_gram::*;

fn main()
{
        let config = Config::default();
        let mut model = Model::new(config);

        // requires `corpus` feature
        let corpus = tiny_corpus()
                .iter()
                .map(|t| sos(eos(tokenize(t.to_owned()))))
                .collect::<Vec<_>>();

        model.train(corpus);

        let mut tokens = sos(tokenize("The quick".to_owned()));
        let max = 10;
        model.generate(&mut tokens, max);

        model.save("model.json").unwrap();
        model.reset();
        model.load("model.json").unwrap();
}
