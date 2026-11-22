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
}

impl Indexable for Memory {
    fn id(&self) -> String {
        self.file.clone()
    }
    fn text(&self) -> String {
        // Field weighting by repetition: name ×3, description ×2, body ×1. A
        // term in the title/description is a stronger signal than the same term
        // buried in the body — the same lever `memory_keyword._doc_tokens`
        // applies on the Python side, which this path had silently dropped.
        format!(
            "{} {} {} {} {} {}",
            self.name, self.name, self.name, self.description, self.description, self.body
        )
    }
}

pub fn bm25(memories: &[Memory], query: &str, limit: usize) -> Vec<Hit> {
    rank(memories, query, limit)
}

/// BM25 over any [`Indexable`] collection.
pub fn rank<T: Indexable>(records: &[T], query: &str, limit: usize) -> Vec<Hit> {
    let documents: Vec<Vec<String>> = records.iter().map(|r| tokenize(&r.text())).collect();
    let terms = tokenize(query);
    if terms.is_empty() || documents.is_empty() {
        return Vec::new();
    }
    let average = documents.iter().map(Vec::len).sum::<usize>() as f64 / documents.len() as f64;
    let mut frequency = HashMap::new();
    for document in &documents {
        for term in document.iter().collect::<HashSet<_>>() {
            *frequency.entry(term.as_str()).or_insert(0usize) += 1;
        }
    }
    let mut hits: Vec<Hit> = documents
        .iter()
        .enumerate()
        .map(|(index, document)| {
            let mut score = 0.0;
            for term in &terms {
                let count = document.iter().filter(|word| *word == term).count() as f64;
                if count == 0.0 {
                    continue;
                }
                let docs = *frequency.get(term.as_str()).unwrap_or(&0) as f64;
                let idf = ((documents.len() as f64 - docs + 0.5) / (docs + 0.5) + 1.0).ln();
                score += idf * count * 2.2
                    / (count + 1.2 * (1.0 - 0.75 + 0.75 * document.len() as f64 / average));
            }
            Hit {
                file: records[index].id(),
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
        .filter(|part| part.len() > 1)
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
const STOPWORDS: &[&str] = &[
    // pt-BR
    "a", "o", "e", "de", "da", "do", "dos", "das", "em", "no", "na", "nos", "nas", "ao", "aos",
    "um", "uma", "uns", "umas", "por", "para", "com", "sem", "que", "quem", "qual", "quais",
    "como", "mas", "se", "meu", "minha", "meus", "minhas", "teu", "tua", "seu", "sua", "seus",
    "suas", "nosso", "nossa", "dele", "dela", "deles", "delas", "ele", "ela", "eles", "elas",
    "voce", "você", "ti", "lhe", "lhes", "foi", "ser", "sao", "esta", "este", "essa", "esse",
    "isso", "isto", "estao", "estão", "está", "tava", "muito", "pouco", "tambem", "também",
    "ainda", "nao", "não", "sim", "quando", "onde", "sobre", "sob", "ate", "até", "pois",
    "porque", "são", "quanto", "quanta", "quantos", "quantas", "eu", "pra", "pro", "pelo",
    "pela", "pelos", "pelas", "num", "dum",
    // en
    "the", "an", "of", "to", "in", "on", "for", "with", "without", "that", "this", "these",
    "those", "it", "its", "is", "are", "was", "were", "been", "being", "have", "has", "had",
    "does", "did", "what", "which", "how", "why", "all", "any", "some", "not", "and", "or",
    "but", "then", "than", "can", "will", "should", "would", "could",
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
        let hits = rrf_weighted(
            &[vec!["a".into()], vec!["b".into()]],
            &[0.0, 1.0],
            2,
            60.0,
        );
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
        assert_eq!(bm25(&memories, "quanto custa a energia do moises de da", 2)[0].file, "bill.md");
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
                body: "stratum proxy configured, stratum proxy defaults, stratum proxy notes".into(),
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
}
