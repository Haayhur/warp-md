//! Atom selection language for filtering structures.
//!
//! Supports VMD-style selection expressions:
//! - `name CA`           — atom name equals CA
//! - `resname ALA`       — residue name equals ALA
//! - `resid 1-10`        — residue sequence IDs 1 through 10
//! - `chain A`           — chain ID A
//! - `protein`           — all 20 standard amino acids
//! - `backbone`          — backbone atoms (N, CA, C, O)
//! - `sidechain`         — non-backbone heavy atoms
//! - `element C`         — element symbol C
//! - `all`               — every atom
//!
//! Combinators: `and`, `or`, `not`, parentheses.
//!
//! Examples:
//! - `chain A and name CA`
//! - `resid 1-10 and backbone`
//! - `not resname GLY`
//! - `(chain A or chain B) and name CA`

use crate::coord::Vec3;
use crate::residue::Structure;

/// Reference to a single atom within a structure.
#[derive(Debug, Clone)]
pub struct AtomRef {
    pub chain_id: char,
    pub resname: String,
    pub resid: i32,
    pub atom_name: String,
    pub element: String,
    pub coord: Vec3,
    pub chain_idx: usize,
    pub res_idx: usize,
    pub atom_idx: usize,
}

/// Parsed selection AST node.
#[derive(Debug, Clone)]
enum Selector {
    All,
    Name(Vec<String>),
    ResName(Vec<String>),
    ResId(Vec<ResIdRange>),
    Chain(Vec<char>),
    Element(Vec<String>),
    Protein,
    Backbone,
    SideChain,
    Not(Box<Selector>),
    And(Box<Selector>, Box<Selector>),
    Or(Box<Selector>, Box<Selector>),
}

#[derive(Debug, Clone)]
struct ResIdRange {
    start: i32,
    end: i32,
}

const BACKBONE_NAMES: &[&str] = &["N", "CA", "C", "O", "OXT"];
const STANDARD_AA: &[&str] = &[
    "GLY", "ALA", "SER", "CYS", "VAL", "ILE", "LEU", "THR", "ARG", "LYS", "ASP", "GLU", "ASN",
    "GLN", "MET", "HIS", "PRO", "PHE", "TYR", "TRP",
];

impl Selector {
    fn matches(&self, a: &AtomRef) -> bool {
        match self {
            Self::All => true,
            Self::Name(names) => names.iter().any(|n| n.eq_ignore_ascii_case(&a.atom_name)),
            Self::ResName(names) => names.iter().any(|n| n.eq_ignore_ascii_case(&a.resname)),
            Self::ResId(ranges) => ranges
                .iter()
                .any(|r| a.resid >= r.start && a.resid <= r.end),
            Self::Chain(ids) => ids.contains(&a.chain_id),
            Self::Element(elems) => elems.iter().any(|e| e.eq_ignore_ascii_case(&a.element)),
            Self::Protein => STANDARD_AA
                .iter()
                .any(|s| s.eq_ignore_ascii_case(&a.resname)),
            Self::Backbone => BACKBONE_NAMES.contains(&a.atom_name.as_str()),
            Self::SideChain => {
                !BACKBONE_NAMES.contains(&a.atom_name.as_str())
                    && !a.element.eq_ignore_ascii_case("H")
            }
            Self::Not(inner) => !inner.matches(a),
            Self::And(l, r) => l.matches(a) && r.matches(a),
            Self::Or(l, r) => l.matches(a) || r.matches(a),
        }
    }
}

/// Apply a selection expression to a structure, returning matching atom refs.
pub fn select(struc: &Structure, expr: &str) -> Result<Vec<AtomRef>, String> {
    let sel = parse_selection(expr)?;
    let mut result = Vec::new();
    for (ci, chain) in struc.chains.iter().enumerate() {
        for (ri, res) in chain.residues.iter().enumerate() {
            let resname = res.amber_name().to_string();
            for (ai, atom) in res.atoms.iter().enumerate() {
                let aref = AtomRef {
                    chain_id: chain.id,
                    resname: resname.clone(),
                    resid: res.seq_id,
                    atom_name: atom.name.clone(),
                    element: atom.element.clone(),
                    coord: atom.coord,
                    chain_idx: ci,
                    res_idx: ri,
                    atom_idx: ai,
                };
                if sel.matches(&aref) {
                    result.push(aref);
                }
            }
        }
    }
    Ok(result)
}

/// Extract just coordinates from a selection.
pub fn select_coords(struc: &Structure, expr: &str) -> Result<Vec<Vec3>, String> {
    Ok(select(struc, expr)?.into_iter().map(|a| a.coord).collect())
}

// ── Parser ──────────────────────────────────────────────────────────────────

fn tokenize(input: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut chars = input.chars().peekable();
    while let Some(&ch) = chars.peek() {
        if ch.is_whitespace() {
            chars.next();
            continue;
        }
        if ch == '(' || ch == ')' {
            tokens.push(ch.to_string());
            chars.next();
            continue;
        }
        let mut word = String::new();
        while let Some(&c) = chars.peek() {
            if c.is_whitespace() || c == '(' || c == ')' {
                break;
            }
            word.push(c);
            chars.next();
        }
        tokens.push(word);
    }
    tokens
}

struct Parser {
    tokens: Vec<String>,
    pos: usize,
}

impl Parser {
    fn new(tokens: Vec<String>) -> Self {
        Self { tokens, pos: 0 }
    }

    fn peek(&self) -> Option<&str> {
        self.tokens.get(self.pos).map(|s| s.as_str())
    }

    fn next(&mut self) -> Option<String> {
        if self.pos < self.tokens.len() {
            let t = self.tokens[self.pos].clone();
            self.pos += 1;
            Some(t)
        } else {
            None
        }
    }

    fn parse_expr(&mut self) -> Result<Selector, String> {
        let left = self.parse_or()?;
        Ok(left)
    }

    fn parse_or(&mut self) -> Result<Selector, String> {
        let mut left = self.parse_and()?;
        while self.peek().map(|t| t.eq_ignore_ascii_case("or")) == Some(true) {
            self.next(); // consume 'or'
            let right = self.parse_and()?;
            left = Selector::Or(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> Result<Selector, String> {
        let mut left = self.parse_not()?;
        while self.peek().map(|t| t.eq_ignore_ascii_case("and")) == Some(true) {
            self.next(); // consume 'and'
            let right = self.parse_not()?;
            left = Selector::And(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_not(&mut self) -> Result<Selector, String> {
        if self.peek().map(|t| t.eq_ignore_ascii_case("not")) == Some(true) {
            self.next();
            let inner = self.parse_primary()?;
            Ok(Selector::Not(Box::new(inner)))
        } else {
            self.parse_primary()
        }
    }

    fn parse_primary(&mut self) -> Result<Selector, String> {
        let tok = self
            .peek()
            .ok_or("unexpected end of selection")?
            .to_lowercase();

        match tok.as_str() {
            "(" => {
                self.next(); // consume '('
                let inner = self.parse_expr()?;
                match self.next() {
                    Some(ref t) if t == ")" => Ok(inner),
                    _ => Err("expected ')'".into()),
                }
            }
            "all" => {
                self.next();
                Ok(Selector::All)
            }
            "protein" => {
                self.next();
                Ok(Selector::Protein)
            }
            "backbone" => {
                self.next();
                Ok(Selector::Backbone)
            }
            "sidechain" => {
                self.next();
                Ok(Selector::SideChain)
            }
            "name" => {
                self.next();
                let vals = self.parse_values()?;
                Ok(Selector::Name(vals))
            }
            "resname" => {
                self.next();
                let vals = self.parse_values()?;
                Ok(Selector::ResName(vals))
            }
            "resid" => {
                self.next();
                let ranges = self.parse_resid_values()?;
                Ok(Selector::ResId(ranges))
            }
            "chain" => {
                self.next();
                let vals = self.parse_values()?;
                let ids: Vec<char> = vals.iter().filter_map(|v| v.chars().next()).collect();
                if ids.is_empty() {
                    return Err("chain requires at least one ID".into());
                }
                Ok(Selector::Chain(ids))
            }
            "element" => {
                self.next();
                let vals = self.parse_values()?;
                Ok(Selector::Element(vals))
            }
            _ => Err(format!("unexpected token '{tok}'")),
        }
    }

    /// Consume one or more non-keyword values (for multi-value: `name CA CB C`).
    fn parse_values(&mut self) -> Result<Vec<String>, String> {
        let mut vals = Vec::new();
        while let Some(t) = self.peek() {
            let low = t.to_lowercase();
            if matches!(
                low.as_str(),
                "and"
                    | "or"
                    | "not"
                    | "("
                    | ")"
                    | "name"
                    | "resname"
                    | "resid"
                    | "chain"
                    | "element"
                    | "all"
                    | "protein"
                    | "backbone"
                    | "sidechain"
            ) {
                break;
            }
            vals.push(self.next().unwrap());
        }
        if vals.is_empty() {
            return Err("expected value after keyword".into());
        }
        Ok(vals)
    }

    fn parse_resid_values(&mut self) -> Result<Vec<ResIdRange>, String> {
        let mut ranges = Vec::new();
        while let Some(t) = self.peek() {
            let low = t.to_lowercase();
            if matches!(
                low.as_str(),
                "and"
                    | "or"
                    | "not"
                    | "("
                    | ")"
                    | "name"
                    | "resname"
                    | "resid"
                    | "chain"
                    | "element"
                    | "all"
                    | "protein"
                    | "backbone"
                    | "sidechain"
            ) {
                break;
            }
            let val = self.next().unwrap();
            // Parse "5" or "1-10" or "1:10"
            if let Some(sep) = val.find('-').or_else(|| val.find(':')) {
                let start: i32 = val[..sep]
                    .parse()
                    .map_err(|_| format!("bad resid '{val}'"))?;
                let end: i32 = val[sep + 1..]
                    .parse()
                    .map_err(|_| format!("bad resid '{val}'"))?;
                ranges.push(ResIdRange { start, end });
            } else {
                let id: i32 = val.parse().map_err(|_| format!("bad resid '{val}'"))?;
                ranges.push(ResIdRange { start: id, end: id });
            }
        }
        if ranges.is_empty() {
            return Err("expected resid value".into());
        }
        Ok(ranges)
    }
}

fn parse_selection(expr: &str) -> Result<Selector, String> {
    let tokens = tokenize(expr);
    if tokens.is_empty() {
        return Ok(Selector::All);
    }
    let mut parser = Parser::new(tokens);
    let sel = parser.parse_expr()?;
    if parser.pos < parser.tokens.len() {
        return Err(format!(
            "unexpected token '{}' at position {}",
            parser.tokens[parser.pos], parser.pos
        ));
    }
    Ok(sel)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::{
        make_extended_structure, make_multi_chain_structure, parse_three_letter_sequence, ChainSpec,
    };

    #[test]
    fn test_select_all() {
        let struc = make_extended_structure("AA").unwrap();
        let atoms = select(&struc, "all").unwrap();
        let total: usize = struc
            .chains
            .iter()
            .map(|c| c.residues.iter().map(|r| r.atoms.len()).sum::<usize>())
            .sum();
        assert_eq!(atoms.len(), total);
    }

    #[test]
    fn test_select_name_ca() {
        let struc = make_extended_structure("AAA").unwrap();
        let atoms = select(&struc, "name CA").unwrap();
        assert_eq!(atoms.len(), 3);
        assert!(atoms.iter().all(|a| a.atom_name == "CA"));
    }

    #[test]
    fn test_select_backbone() {
        let struc = make_extended_structure("AG").unwrap();
        let bb = select(&struc, "backbone").unwrap();
        // ALA: N,CA,C,O + CB(sidechain) = 4 backbone, GLY: N,CA,C,O = 4 backbone
        assert_eq!(bb.len(), 8);
    }

    #[test]
    fn test_select_resid_range() {
        let struc = make_extended_structure("AAAA").unwrap();
        let atoms = select(&struc, "resid 2-3 and name CA").unwrap();
        assert_eq!(atoms.len(), 2);
    }

    #[test]
    fn test_select_chain() {
        let chains = vec![
            ChainSpec {
                id: 'A',
                residues: parse_three_letter_sequence("ALA-ALA").unwrap(),
                preset: None,
            },
            ChainSpec {
                id: 'B',
                residues: parse_three_letter_sequence("GLY-GLY").unwrap(),
                preset: None,
            },
        ];
        let struc = make_multi_chain_structure(&chains).unwrap();
        let atoms = select(&struc, "chain B and name CA").unwrap();
        assert_eq!(atoms.len(), 2);
        assert!(atoms.iter().all(|a| a.chain_id == 'B'));
    }

    #[test]
    fn test_select_not() {
        let struc = make_extended_structure("AG").unwrap();
        let not_gly = select(&struc, "not resname GLY").unwrap();
        assert!(not_gly.iter().all(|a| a.resname != "GLY"));
    }

    #[test]
    fn test_select_or() {
        let struc = make_extended_structure("ACDE").unwrap();
        let atoms = select(&struc, "resname ALA or resname CYS").unwrap();
        assert!(atoms
            .iter()
            .all(|a| a.resname == "ALA" || a.resname == "CYS"));
    }

    #[test]
    fn test_select_parens() {
        let struc = make_extended_structure("ACDE").unwrap();
        let atoms = select(&struc, "(resname ALA or resname CYS) and name CA").unwrap();
        assert_eq!(atoms.len(), 2);
    }

    #[test]
    fn test_select_protein() {
        let struc = make_extended_structure("AA").unwrap();
        let atoms = select(&struc, "protein").unwrap();
        let total: usize = struc
            .chains
            .iter()
            .map(|c| c.residues.iter().map(|r| r.atoms.len()).sum::<usize>())
            .sum();
        assert_eq!(atoms.len(), total);
    }

    #[test]
    fn test_select_sidechain() {
        let struc = make_extended_structure("A").unwrap();
        let sc = select(&struc, "sidechain").unwrap();
        // ALA sidechain = CB only
        assert_eq!(sc.len(), 1);
        assert_eq!(sc[0].atom_name, "CB");
    }

    #[test]
    fn test_select_empty_string() {
        let struc = make_extended_structure("A").unwrap();
        let atoms = select(&struc, "").unwrap();
        assert!(!atoms.is_empty()); // empty = all
    }

    #[test]
    fn test_select_error() {
        let struc = make_extended_structure("A").unwrap();
        assert!(select(&struc, "bogus").is_err());
    }

    #[test]
    fn test_select_coords() {
        let struc = make_extended_structure("AA").unwrap();
        let coords = select_coords(&struc, "name CA").unwrap();
        assert_eq!(coords.len(), 2);
    }

    #[test]
    fn test_select_multi_name() {
        let struc = make_extended_structure("A").unwrap();
        let atoms = select(&struc, "name CA CB").unwrap();
        assert_eq!(atoms.len(), 2);
    }

    #[test]
    fn test_select_element() {
        let struc = make_extended_structure("C").unwrap();
        let sulfurs = select(&struc, "element S").unwrap();
        assert_eq!(sulfurs.len(), 1);
        assert_eq!(sulfurs[0].atom_name, "SG");
    }
}
