use engram_store::Memory;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug, PartialEq)]
pub struct Hit {
    pub file: String,
    pub score: f64,
}

pub fn rrf(rankings: &[Vec<String>], limit: usize, k: f64) -> Vec<Hit> {
    let weights: Vec<f64> = rankings.iter().map(|_| 1.0).collect();
    rrf_weighted(rankings, &weights, limit, k)
}

/// RRF with a per-ranker weight. `engram.yaml`'s `recall.hybrid.weights` is
/// documented on both sides and honoured by the Python fusion; the Rust path
/// had no parameter for it, so the config key was silently inert here.
pub fn rrf_weighted(rankings: &[Vec<String>], weights: &[f64], limit: usize, k: f64) -> Vec<Hit> {
    debug_assert_eq!(rankings.len(), weights.len(), "one weight per ranking");
    let mut scores: HashMap<String, f64> = HashMap::new();
    for (ranking, weight) in rankings.iter().zip(weights) {
        for (position, file) in ranking.iter().enumerate() {
            *scores.entry(file.clone()).or_default() += weight / (k + position as f64 + 1.0);
        }
    }
    let mut hits: Vec<Hit> = scores
        .into_iter()
        .map(|(file, score)| Hit { file, score })
        .collect();
    hits.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.file.cmp(&right.file))
    });
    hits.truncate(limit);
    hits
}

/// Anything BM25 can rank: an id to return, and the text to score.
///
/// Introduced so wiki chunks reuse the ranking rather than getting a second,
/// subtly different implementation. `Memory` implements it and the memory path
/// is unchanged — `bm25` is still the entry point for it.
pub trait Indexable {
    /// What the hit identifies. A memory filename, or `path#chunk` for a chunk.
    fn id(&self) -> String;
    /// Everything searchable about the record, concatenated.
    fn text(&self) -> String;
    /// (weight, text) per field. A term in a NAME or a DESCRIPTION is stronger
    /// evidence that this record IS the thing the query asks about than the same
    /// term in a body that merely mentions it, so the weight multiplies that term's
    /// idf. Records without fields — a wiki chunk — take the default.
    ///
    /// This replaces the old trick of repeating name ×3 and description ×2 into one
    /// string. That trick does nothing under BM25: tf saturation (k1 = 1.2) flattens
    /// tf 3 to within a few percent of tf 1, so the field weighting was inert — and
    /// the repetition inflated avgdl, which weakened length normalization for every
    /// other document in the corpus.
    fn fields(&self) -> Vec<(f64, String)> {
        vec![(1.0, self.text())]
    }
}

impl Indexable for Memory {
    fn id(&self) -> String {
        self.file.clone()
    }
    fn text(&self) -> String {
        format!("{} {} {}", self.name, self.description, self.body)
    }
    fn fields(&self) -> Vec<(f64, String)> {
        vec![
            (3.0, self.name.clone()),
            (2.0, self.description.clone()),
            (1.0, self.body.clone()),
        ]
    }
}

pub fn bm25(memories: &[Memory], query: &str, limit: usize) -> Vec<Hit> {
    rank(memories, query, limit)
}

/// One record as the ranker sees it: its length (for BM25's length normalization),
/// the term frequencies (tf and df), and the strongest field weight each term
/// earned. Only the counts are kept as tokens — the token stream itself is never
/// read again, and cloning 500 tokens per record x 895 records per prompt is the
/// whole cost of keeping it.
struct Ranked {
    id: String,
    len: usize,
    counts: HashMap<String, usize>,
    weight: HashMap<String, f64>,
}

impl Ranked {
    fn new<T: Indexable>(record: &T) -> Self {
        let mut ranked = Self {
            id: record.id(),
            len: 0,
            counts: HashMap::new(),
            weight: HashMap::new(),
        };
        for (weight, text) in record.fields() {
            for term in tokenize(&text) {
                ranked.len += 1;
                *ranked.counts.entry(term.clone()).or_insert(0) += 1;
                let best = ranked.weight.entry(term).or_insert(0.0);
                *best = best.max(weight);
            }
        }
        ranked
    }
}

/// BM25 over any [`Indexable`] collection.
pub fn rank<T: Indexable>(records: &[T], query: &str, limit: usize) -> Vec<Hit> {
    let documents: Vec<Ranked> = records.iter().map(Ranked::new).collect();
    // Deduped, in query order: a word the user happened to type twice is one
    // signal, not two, and a stable order keeps the score summable bit-for-bit.
    let mut terms: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for term in tokenize(query) {
        if seen.insert(term.clone()) {
            terms.push(term);
        }
    }
    if terms.is_empty() || documents.is_empty() {
        return Vec::new();
    }
    let average = documents.iter().map(|d| d.len).sum::<usize>() as f64 / documents.len() as f64;
    let mut frequency: HashMap<&str, usize> = HashMap::new();
    for document in &documents {
        // Counted from the term map, so a word repeated 40 times in one body
        // does not make itself look rare.
        for term in document.counts.keys() {
            *frequency.entry(term.as_str()).or_insert(0usize) += 1;
        }
    }
    let mut hits: Vec<Hit> = documents
        .iter()
        .map(|document| {
            let mut score = 0.0;
            for term in &terms {
                let count = *document.counts.get(term.as_str()).unwrap_or(&0) as f64;
                if count == 0.0 {
                    continue;
                }
                let docs = *frequency.get(term.as_str()).unwrap_or(&0) as f64;
                let idf = ((documents.len() as f64 - docs + 0.5) / (docs + 0.5) + 1.0).ln();
                let field = *document.weight.get(term.as_str()).unwrap_or(&1.0);
                score += field * idf * count * 2.2
                    / (count + 1.2 * (1.0 - 0.75 + 0.75 * document.len as f64 / average));
            }
            Hit {
                file: document.id.clone(),
                score,
            }
        })
        .filter(|hit| hit.score > 0.0)
        .collect();
    hits.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.file.cmp(&right.file))
    });
    hits.truncate(limit);
    hits
}

fn tokenize(text: &str) -> Vec<String> {
    text.split(|ch: char| !ch.is_alphanumeric() && ch != '_' && ch != '-')
        // Two characters or more, counted in CHARS: `len()` counts bytes, so "é"
        // and "à" used to pass here, fold to "e"/"a", and then vote as if they
        // were the common function words they are.
        .filter(|part| part.chars().nth(1).is_some())
        .flat_map(|part| {
            // A compound id is indexed BOTH ways: whole, so an exact id query
            // ("api-1", "pdv-relay") still lands the file that literally contains
            // it, and by its parts, so "robosats" matches the memory NAMED
            // `reference-robosats-gateway`. Emitting only the whole token — what
            // this did — threw away the name and description weight for every
            // kebab/snake name in the store, i.e. the whole point of the field
            // weighting above: a query for the topic word could only ever match
            // bodies, and an unrelated document that happened to use a
            // conversational word outranked the memory the query names.
            //
            // This matches `bin/memory_keyword.py` for '_' and '-'. It does NOT
            // match it for '.' '/' ':', which THIS tokenizer has always split
            // ("127.0.0.1:6333" → "127" "0" "0" "1" "6333") while Python joins
            // them into one token. Unchanged here, because nothing measured this
            // afternoon depended on it; it is a known parity gap, not a new one.
            let mut words = vec![part];
            if part.contains(['_', '-']) {
                words.extend(
                    part.split(['_', '-'])
                        .filter(|inner| inner.chars().nth(1).is_some()),
                );
            }
            words
        })
        .filter_map(|part| {
            // Check the stopword BEFORE trusting the fold, and exempt acronyms
            // written in full capitals: "DAS" (the Simples guia), "NAS", "SIM"
            // are content in this store even though their lowercase forms are
            // function words. The exemption compares against the ORIGINAL
            // part, before lowercasing.
            let acronym = part.chars().all(char::is_uppercase);
            let lower = part.to_lowercase();
            // Stopword check runs on the pre-fold form (see STOPWORDS):
            // "nó" must survive as content, and the accented function words
            // are in the list itself.
            if acronym || !is_stopword(&lower) {
                Some(fold_diacritics(&lower))
            } else {
                None
            }
        })
        .collect()
}

/// Function words for the two languages this store is written in (pt-BR and
/// en). Deliberately function words only: "mes", "ano", "energia" carry
/// meaning in real queries and must never be swallowed here.
///
/// Entries match the LOWERCASED word, before diacritic folding — on purpose.
/// "nó" (a Lightning node) and "nós" must survive as content even though
/// they fold to "no"/"nos", so the accented FUNCTION words ("não", "são"…)
/// are listed explicitly instead of relying on the fold. Both unaccented and
/// accented shapes are listed: people type prompts without accents.
#[rustfmt::skip]
const STOPWORDS: &[&str] = &[
    // pt-BR.
    "a", "o", "e", "de", "da", "do", "dos", "das", "em", "no", "na", "nos", "nas", "ao", "aos",
    "um", "uma", "uns", "umas", "por", "para", "com", "sem", "que", "quem", "qual", "quais",
    "como", "mas", "se", "meu", "minha", "meus", "minhas", "teu", "tua", "seu", "sua", "seus",
    "suas", "nosso", "nossa", "dele", "dela", "deles", "delas", "ele", "ela", "eles", "elas",
    "voce", "você", "ti", "lhe", "lhes", "foi", "ser", "sao", "esta", "este", "essa", "esse",
    "isso", "isto", "estao", "estão", "está", "tava", "muito", "pouco", "tambem", "também",
    "ainda", "nao", "não", "sim", "quando", "onde", "sobre", "sob", "ate", "até", "pois",
    "porque", "são", "quanto", "quanta", "quantos", "quantas", "eu", "pra", "pro", "pelo",
    "pela", "pelos", "pelas", "num", "dum",
    // pt-BR, second rank — added 2026-11-24 after measuring a real false injection: the
    // operator's prompt about store languages injected a memory about his mother and one about
    // tailscale/ufw. The tailscale memory shared nothing topical with that prompt; it matched it
    // through as, ou, entao, seja, todas — grammar, not topic (plus `caso`, which stays content:
    // it is also a noun, and it was not needed once its neighbours left). Then the auxiliaries
    // and modals: a verb in a prompt says what the user wants done, never what the memory is
    // about.
    "as", "os", "todo", "toda", "todos", "todas", "tudo", "cada", "algum", "alguma",
    "ou", "entao", "então", "assim",
    "seja", "sejam", "estava", "estavam", "ter", "tem", "temos", "tinham", "deve", "devem",
    "dever", "deveria", "preciso", "precisa", "precisamos", "posso", "pode", "podem",
    "poderia", "quer", "quero", "queria", "gostaria", "usa", "usam", "usar", "faz", "faco",
    "faço", "fazer", "conter", "contem", "contém", "aplica", "aplicam", "aplicar",
    // One-character function words that arrive ACCENTED: the length filter counts chars (it
    // counted bytes until PR #50), and these fold to "e"/"a"/"ja"/"ha" — so they are listed in
    // their real shape, per the contract above. "já"/"há" are plain function words in this
    // store's prompts ("o que já temos pronto"). The rest are the ACCENTED TWINS of forms the
    // lists above already carry: the stopword test runs on the pre-fold word, so "quê" does not
    // match the "que" entry and "dê" does not match "de". Unlisted, each one folds into a term
    // the unaccented shape would have dropped — and since the unaccented spelling is dropped from
    // EVERY document, that term survives only in the few memories that spell it accented: tiny
    // df, and one of the strongest idf votes in the corpus. That is worse than not listing it.
    // "nós" is NOT here, on purpose: the contract above says it survives as content, and in this
    // store it means nodes (headscale, tailnet, Lightning), not "we".
    "é", "à", "às", "há", "já", "ja", "ha", "só", "têm", "dá", "quê", "porquê", "dê",
    // en.
    "the", "an", "of", "to", "in", "on", "for", "with", "without", "that", "this", "these",
    "those", "it", "its", "is", "are", "was", "were", "been", "being", "have", "has", "had",
    "does", "did", "what", "which", "how", "why", "all", "any", "some", "not", "and", "or",
    "but", "then", "than", "can", "will", "should", "would", "could",
    // en, second rank: pronouns, copulas, modals. Measured the same way — an English prompt's
    // surviving terms included we, much, want, my, about.
    "we", "you", "your", "yours", "my", "mine", "our", "ours", "their", "they", "them",
    "him", "his", "her", "hers", "us", "me", "be", "am", "if", "so", "about", "into",
    "from", "at", "there", "here", "just", "much", "want", "wants", "need", "needs",
    "already", "cannot", "cant", "dont", "wont",
];

pub fn is_stopword(word: &str) -> bool {
    let lower = word.to_lowercase();
    STOPWORDS.contains(&lower.as_str())
}

/// Fold Latin diacritics to ASCII, after lowercasing. The store and the
/// queries are the same language — Portuguese written WITH accents in the
/// memories, and users type prompts without them (`moises`, `mineracao` vs
/// `Moisés`, `mineração`) — and an un-folded token simply never matches.
/// A table, not NFD: std has no normalization, and these are the accented
/// letters these stores actually contain.
/// ponytail: table-based; reach for unicode-normalization if other scripts show up.
fn fold_diacritics(word: &str) -> String {
    word.chars()
        .map(|c| match c {
            'á' | 'à' | 'â' | 'ä' | 'ã' | 'å' | 'ā' | 'ă' | 'ą' | 'æ' => 'a',
            'ç' | 'ć' | 'č' | 'ĉ' => 'c',
            'é' | 'è' | 'ê' | 'ë' | 'ē' | 'ĕ' | 'ė' | 'ę' | 'ě' => 'e',
            'í' | 'ì' | 'î' | 'ï' | 'ī' | 'ĭ' | 'į' | 'ı' => 'i',
            'ð' => 'd',
            'ñ' | 'ń' | 'ň' | 'ņ' => 'n',
            'ó' | 'ò' | 'ô' | 'ö' | 'õ' | 'ø' | 'ō' | 'ő' | 'œ' => 'o',
            'ř' => 'r',
            'š' | 'ş' | 'ś' => 's',
            'ť' | 'ț' => 't',
            'ú' | 'ù' | 'û' | 'ü' | 'ū' | 'ŭ' | 'ů' | 'ű' | 'ų' => 'u',
            'ý' | 'ÿ' => 'y',
            'ž' | 'ź' | 'ż' => 'z',
            'ß' => 's',
            other => other,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn finds_exact_infrastructure_terms() {
        let memories = vec![
            Memory {
                file: "dns.md".into(),
                name: "DNS".into(),
                description: "AdGuard ULA".into(),
                memory_type: "reference".into(),
                body: "IPv6 DNS server".into(),
                source_mtime: 0,
            },
            Memory {
                file: "other.md".into(),
                name: "Other".into(),
                description: "unrelated".into(),
                memory_type: "reference".into(),
                body: "".into(),
                source_mtime: 0,
            },
        ];
        assert_eq!(bm25(&memories, "IPv6 DNS", 1)[0].file, "dns.md");
    }
    #[test]
    fn rrf_rewards_consensus() {
        assert_eq!(
            rrf(&[vec!["a".into(), "b".into()], vec!["b".into()]], 2, 60.0)[0].file,
            "b"
        );
    }
    #[test]
    fn rrf_weight_can_demote_a_leg() {
        // "a" leads the first ranking; a weight of 0 on that ranking must hand
        // the win to "b", which the second ranking put first.
        let hits = rrf_weighted(&[vec!["a".into()], vec!["b".into()]], &[0.0, 1.0], 2, 60.0);
        assert_eq!(hits[0].file, "b");
    }
    #[test]
    fn unaccented_queries_match_accented_text() {
        // The pt-BR case that started this: a prompt says "mineracao moises",
        // the memory says "mineração" and "Moisés".
        let memories = vec![
            Memory {
                file: "mineracao.md".into(),
                name: "mineração".into(),
                description: "Moisés hosting".into(),
                memory_type: "project".into(),
                body: "energia viabtc".into(),
                source_mtime: 0,
            },
            Memory {
                file: "other.md".into(),
                name: "Other".into(),
                description: "unrelated".into(),
                memory_type: "project".into(),
                body: "".into(),
                source_mtime: 0,
            },
        ];
        let hits = bm25(&memories, "mineracao moises", 2);
        assert_eq!(hits[0].file, "mineracao.md");
        // And the other direction: an accented query must match plain text.
        assert_eq!(bm25(&memories, "mineração", 1)[0].file, "mineracao.md");
    }
    #[test]
    fn stopwords_do_not_drive_ranking() {
        // The real failure: a TeamSpeak memory that merely says "quanto custa"
        // beat the energy-bill memory that the query is about.
        let memories = vec![
            Memory {
                file: "noise.md".into(),
                name: "TeamSpeak notify".into(),
                description: "reverted".into(),
                memory_type: "project".into(),
                body: "quanto custa de da do quanto custa de da do".into(),
                source_mtime: 0,
            },
            Memory {
                file: "bill.md".into(),
                name: "moises energy billing".into(),
                description: "moises energia bill".into(),
                memory_type: "reference".into(),
                body: "energia".into(),
                source_mtime: 0,
            },
        ];
        assert_eq!(
            bm25(&memories, "quanto custa a energia do moises de da", 2)[0].file,
            "bill.md"
        );
    }
    #[test]
    fn a_compound_name_is_searchable_whole_and_by_its_parts() {
        // The root cause, asserted directly instead of through a ranking: the
        // memory NAMES in this store are kebab/snake ids, and if the whole name is
        // one token the topic word never reaches the field the ranker weights most.
        let words = tokenize("reference-robosats-gateway");
        assert!(words.contains(&"reference-robosats-gateway".to_string()));
        assert!(words.contains(&"robosats".to_string()));
        // Exact ids still match whole ("api-1" is the documented case) and a
        // single-character part is not emitted.
        let words = tokenize("api-1");
        assert!(words.contains(&"api-1".to_string()));
        assert!(words.contains(&"api".to_string()));
        assert!(!words.contains(&"1".to_string()));
    }

    #[test]
    fn a_memory_named_for_the_topic_beats_prose_that_merely_uses_it() {
        // The owner's failure, in two documents. "robosats" appears ONLY inside the
        // kebab name — nowhere else — so the old tokenizer could not see it at all
        // and the prompt's conversational words won.
        let memories = vec![
            Memory {
                file: "reference-robosats-gateway.md".into(),
                name: "reference-robosats-gateway".into(),
                description: "P2P swap coordinator on the gateway".into(),
                memory_type: "reference".into(),
                body: "coordinator onion, rs-tn wrapper".into(),
                source_mtime: 0,
            },
            Memory {
                file: "channel-export.md".into(),
                name: "channel export".into(),
                description: "how to export our own channel".into(),
                memory_type: "feedback".into(),
                body: "vamos continuar com o projeto quando houver tempo".into(),
                source_mtime: 0,
            },
        ];
        let hits = bm25(&memories, "vamos continuar com robosats coordinator", 2);
        assert_eq!(hits[0].file, "reference-robosats-gateway.md");
    }

    #[test]
    fn a_repeated_query_word_is_one_signal_not_two() {
        let memories = vec![
            Memory {
                file: "a.md".into(),
                name: "stratum".into(),
                description: "stratum".into(),
                memory_type: "project".into(),
                body: "one mention".into(),
                source_mtime: 0,
            },
            Memory {
                file: "b.md".into(),
                name: "other".into(),
                description: "unrelated".into(),
                memory_type: "project".into(),
                body: "unrelated".into(),
                source_mtime: 0,
            },
        ];
        // "stratum" once and twice must rank "a" first by the same margin: the
        // duplicate must not double its own term's weight.
        let once = bm25(&memories, "stratum other", 2);
        let twice = bm25(&memories, "stratum stratum other", 2);
        assert_eq!(once[0].file, "a.md");
        assert_eq!(twice[0].file, "a.md");
        let ratio = |hits: &Vec<Hit>| hits[0].score / hits[1].score;
        assert!(
            (ratio(&once) - ratio(&twice)).abs() < 0.01,
            "duplicate changed the ranking: {once:?} vs {twice:?}"
        );
    }

    #[test]
    fn title_matches_outrank_body_matches() {
        // Same term, twice in the corpus: the memory that has it in the NAME
        // must rank above the one that only buries it in the body.
        let memories = vec![
            Memory {
                file: "bodyonly.md".into(),
                name: "Kiosk panel".into(),
                description: "screen".into(),
                memory_type: "project".into(),
                body: "stratum proxy configured, stratum proxy defaults, stratum proxy notes"
                    .into(),
                source_mtime: 0,
            },
            Memory {
                file: "titled.md".into(),
                name: "stratum proxy".into(),
                description: "stratum endpoint on the vps".into(),
                memory_type: "project".into(),
                body: "one mention of stratum".into(),
                source_mtime: 0,
            },
        ];
        assert_eq!(bm25(&memories, "stratum proxy", 2)[0].file, "titled.md");
    }

    #[test]
    fn swallows_second_rank_function_words() {
        // Measured 2026-11-24 on the live store: a prompt about store languages
        // injected a tailscale/ufw memory that matched ONLY these words. Grammar is
        // not a topic, and a word that appears in almost every prompt is the least
        // informative thing a document can share with a query.
        let grammar = "as os ou entao então assim todas todos tudo cada seja tem temos deve \
                       devem precisa queria usam fazer muito much about their you your already \
                       só têm dá quê porquê dê";
        for word in grammar.split_whitespace() {
            assert!(is_stopword(word), "{word} must be a function word");
        }
        // Never a topic word — including `mae`, the high-idf word (4 of 1133 memories)
        // that pulled a memory about the owner's mother into a systems prompt, and
        // `caso`, deliberately left out because it is also a noun. `nó`/`nós` are the store's
        // own words for its nodes (headscale, tailnet, Lightning): 12 memories use them.
        let content = "mae mãe caso engram energia robosats vaultwarden espanhol lingua fatura \
                       nó nós";
        for word in content.split_whitespace() {
            assert!(!is_stopword(word), "{word} is content");
        }
    }
}
