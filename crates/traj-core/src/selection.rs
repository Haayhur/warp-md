use std::sync::Arc;

use crate::error::{TrajError, TrajResult};
use crate::system::System;

#[derive(Debug, Clone)]
pub struct Selection {
    pub expr: String,
    pub indices: Arc<Vec<u32>>,
}

#[derive(Debug, Clone)]
enum SelExpr {
    And(Box<SelExpr>, Box<SelExpr>),
    Or(Box<SelExpr>, Box<SelExpr>),
    Not(Box<SelExpr>),
    Pred(Predicate),
}

#[derive(Debug, Clone)]
enum Predicate {
    Name(u32),
    Resname(u32),
    ResidRange(i32, i32),
    Chain(u32),
    Protein,
    Backbone,
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Ident(String),
    Int(i32),
    Colon,
    LParen,
    RParen,
    Eof,
}

struct Lexer {
    input: Vec<char>,
    pos: usize,
}

impl Lexer {
    fn new(input: &str) -> Self {
        Self {
            input: input.chars().collect(),
            pos: 0,
        }
    }

    fn next_token(&mut self) -> TrajResult<Token> {
        self.skip_ws();
        let ch = match self.peek() {
            Some(c) => c,
            None => return Ok(Token::Eof),
        };
        match ch {
            ':' => {
                self.pos += 1;
                Ok(Token::Colon)
            }
            '(' => {
                self.pos += 1;
                Ok(Token::LParen)
            }
            ')' => {
                self.pos += 1;
                Ok(Token::RParen)
            }
            '-' | '0'..='9' => self.lex_number(),
            _ if is_ident_start(ch) => self.lex_ident(),
            _ => Err(TrajError::InvalidSelection(format!(
                "unexpected character '{ch}'",
            ))),
        }
    }

    fn skip_ws(&mut self) {
        while let Some(c) = self.peek() {
            if c.is_whitespace() {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn peek(&self) -> Option<char> {
        self.input.get(self.pos).copied()
    }

    fn lex_ident(&mut self) -> TrajResult<Token> {
        let start = self.pos;
        while let Some(c) = self.peek() {
            if is_ident_continue(c) {
                self.pos += 1;
            } else {
                break;
            }
        }
        let s: String = self.input[start..self.pos].iter().collect();
        Ok(Token::Ident(s))
    }

    fn lex_number(&mut self) -> TrajResult<Token> {
        let start = self.pos;
        if self.peek() == Some('-') {
            self.pos += 1;
        }
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() {
                self.pos += 1;
            } else {
                break;
            }
        }
        let s: String = self.input[start..self.pos].iter().collect();
        let val: i32 = s
            .parse()
            .map_err(|_| TrajError::InvalidSelection(format!("invalid integer '{s}'")))?;
        Ok(Token::Int(val))
    }
}

fn is_ident_start(c: char) -> bool {
    c.is_ascii_alphabetic()
}

fn is_ident_continue(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_' || c == '.' || c == '*'
}

struct Parser<'a> {
    lexer: Lexer,
    lookahead: Token,
    system: &'a mut System,
}

impl<'a> Parser<'a> {
    fn new(input: &str, system: &'a mut System) -> TrajResult<Self> {
        let mut lexer = Lexer::new(input);
        let lookahead = lexer.next_token()?;
        Ok(Self {
            lexer,
            lookahead,
            system,
        })
    }

    fn bump(&mut self) -> TrajResult<Token> {
        let current = std::mem::replace(&mut self.lookahead, Token::Eof);
        self.lookahead = self.lexer.next_token()?;
        Ok(current)
    }

    fn expect_ident(&mut self) -> TrajResult<String> {
        match self.bump()? {
            Token::Ident(s) => Ok(s),
            other => Err(TrajError::InvalidSelection(format!(
                "expected identifier, got {other:?}",
            ))),
        }
    }

    fn expect_int(&mut self) -> TrajResult<i32> {
        match self.bump()? {
            Token::Int(v) => Ok(v),
            other => Err(TrajError::InvalidSelection(format!(
                "expected integer, got {other:?}",
            ))),
        }
    }

    fn parse(&mut self) -> TrajResult<SelExpr> {
        let expr = self.parse_or()?;
        match self.lookahead {
            Token::Eof => Ok(expr),
            _ => Err(TrajError::InvalidSelection(
                "unexpected tokens at end of selection".into(),
            )),
        }
    }

    fn parse_or(&mut self) -> TrajResult<SelExpr> {
        let mut left = self.parse_and()?;
        loop {
            match &self.lookahead {
                Token::Ident(ident) if ident.eq_ignore_ascii_case("or") => {
                    self.bump()?;
                    let right = self.parse_and()?;
                    left = SelExpr::Or(Box::new(left), Box::new(right));
                }
                _ => break,
            }
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> TrajResult<SelExpr> {
        let mut left = self.parse_not()?;
        loop {
            match &self.lookahead {
                Token::Ident(ident) if ident.eq_ignore_ascii_case("and") => {
                    self.bump()?;
                    let right = self.parse_not()?;
                    left = SelExpr::And(Box::new(left), Box::new(right));
                }
                _ => break,
            }
        }
        Ok(left)
    }

    fn parse_not(&mut self) -> TrajResult<SelExpr> {
        match &self.lookahead {
            Token::Ident(ident) if ident.eq_ignore_ascii_case("not") => {
                self.bump()?;
                let expr = self.parse_not()?;
                Ok(SelExpr::Not(Box::new(expr)))
            }
            _ => self.parse_primary(),
        }
    }

    fn parse_primary(&mut self) -> TrajResult<SelExpr> {
        match &self.lookahead {
            Token::LParen => {
                self.bump()?;
                let expr = self.parse_or()?;
                match self.bump()? {
                    Token::RParen => Ok(expr),
                    other => Err(TrajError::InvalidSelection(format!(
                        "expected ')', got {other:?}",
                    ))),
                }
            }
            _ => self.parse_predicate(),
        }
    }

    fn parse_predicate(&mut self) -> TrajResult<SelExpr> {
        let key = self.expect_ident()?;
        let lower = key.to_ascii_lowercase();
        let pred = match lower.as_str() {
            "name" => {
                let name = self.expect_ident()?;
                let id = self.system.interner.intern_upper(&name);
                Predicate::Name(id)
            }
            "resname" => {
                let resname = self.expect_ident()?;
                let id = self.system.interner.intern_upper(&resname);
                Predicate::Resname(id)
            }
            "resid" => {
                let start = self.expect_int()?;
                if let Token::Colon = self.lookahead {
                    self.bump()?;
                    let end = self.expect_int()?;
                    Predicate::ResidRange(start, end)
                } else {
                    Predicate::ResidRange(start, start)
                }
            }
            "chain" => {
                let chain = self.expect_ident()?;
                let id = self.system.interner.intern_upper(&chain);
                Predicate::Chain(id)
            }
            "protein" => Predicate::Protein,
            "backbone" => Predicate::Backbone,
            _ => {
                return Err(TrajError::InvalidSelection(format!(
                    "unknown predicate '{key}'",
                )))
            }
        };
        Ok(SelExpr::Pred(pred))
    }
}

pub fn compile_selection(expr: &str, system: &mut System) -> TrajResult<Selection> {
    let mut parser = Parser::new(expr, system)?;
    let ast = parser.parse()?;
    let mask = eval(&ast, system)?;
    let mut indices = Vec::new();
    for (i, &keep) in mask.iter().enumerate() {
        if keep {
            indices.push(i as u32);
        }
    }
    Ok(Selection {
        expr: expr.to_string(),
        indices: Arc::new(indices),
    })
}

fn eval(expr: &SelExpr, system: &System) -> TrajResult<Vec<bool>> {
    match expr {
        SelExpr::Pred(pred) => eval_predicate(pred, system),
        SelExpr::And(a, b) => {
            let left = eval(a, system)?;
            let right = eval(b, system)?;
            Ok(and_mask(&left, &right))
        }
        SelExpr::Or(a, b) => {
            let left = eval(a, system)?;
            let right = eval(b, system)?;
            Ok(or_mask(&left, &right))
        }
        SelExpr::Not(a) => {
            let left = eval(a, system)?;
            Ok(not_mask(&left))
        }
    }
}

fn eval_predicate(pred: &Predicate, system: &System) -> TrajResult<Vec<bool>> {
    let n = system.n_atoms();
    let mut mask = vec![false; n];
    match pred {
        Predicate::Name(id) => {
            for i in 0..n {
                if system.atoms.name_id[i] == *id {
                    mask[i] = true;
                }
            }
        }
        Predicate::Resname(id) => {
            for i in 0..n {
                if system.atoms.resname_id[i] == *id {
                    mask[i] = true;
                }
            }
        }
        Predicate::ResidRange(start, end) => {
            let (lo, hi) = if start <= end {
                (*start, *end)
            } else {
                (*end, *start)
            };
            for i in 0..n {
                let resid = system.atoms.resid[i];
                if resid >= lo && resid <= hi {
                    mask[i] = true;
                }
            }
        }
        Predicate::Chain(id) => {
            for i in 0..n {
                if system.atoms.chain_id[i] == *id {
                    mask[i] = true;
                }
            }
        }
        Predicate::Protein => {
            for i in 0..n {
                let resname_id = system.atoms.resname_id[i];
                let resname = system.interner.resolve(resname_id).unwrap_or("");
                if is_protein_resname(resname) {
                    mask[i] = true;
                }
            }
        }
        Predicate::Backbone => {
            for i in 0..n {
                let resname_id = system.atoms.resname_id[i];
                let name_id = system.atoms.name_id[i];
                let resname = system.interner.resolve(resname_id).unwrap_or("");
                let name = system.interner.resolve(name_id).unwrap_or("");
                if is_protein_resname(resname) && is_backbone_name(name) {
                    mask[i] = true;
                }
            }
        }
    }
    Ok(mask)
}

fn and_mask(a: &[bool], b: &[bool]) -> Vec<bool> {
    a.iter().zip(b).map(|(x, y)| *x && *y).collect()
}

fn or_mask(a: &[bool], b: &[bool]) -> Vec<bool> {
    a.iter().zip(b).map(|(x, y)| *x || *y).collect()
}

fn not_mask(a: &[bool]) -> Vec<bool> {
    a.iter().map(|x| !*x).collect()
}

fn is_protein_resname(name: &str) -> bool {
    let upper = name.to_ascii_uppercase();
    PROTEIN_RESNAMES.contains(&upper.as_str())
}

fn is_backbone_name(name: &str) -> bool {
    let upper = name.to_ascii_uppercase();
    BACKBONE_NAMES.contains(&upper.as_str())
}

const PROTEIN_RESNAMES: &[&str] = &[
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET",
    "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "MSE", "HSD", "HSE", "HSP",
];

const BACKBONE_NAMES: &[&str] = &["N", "CA", "C", "O", "OXT"];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interner::StringInterner;
    use crate::system::{AtomTable, System};

    fn build_system() -> System {
        let mut interner = StringInterner::new();
        let name_n = interner.intern_upper("N");
        let name_ca = interner.intern_upper("CA");
        let name_cb = interner.intern_upper("CB");
        let res_ala = interner.intern_upper("ALA");
        let res_hoh = interner.intern_upper("HOH");
        let chain_a = interner.intern_upper("A");
        let chain_b = interner.intern_upper("B");
        let atoms = AtomTable {
            name_id: vec![name_n, name_ca, name_cb, name_n],
            resname_id: vec![res_ala, res_ala, res_ala, res_hoh],
            resid: vec![1, 1, 1, 2],
            chain_id: vec![chain_a, chain_a, chain_a, chain_b],
            element_id: vec![0, 0, 0, 0],
            mass: vec![1.0, 1.0, 1.0, 1.0],
        };
        System::with_atoms(atoms, interner, None)
    }

    #[test]
    fn selection_name() {
        let mut system = build_system();
        let sel = compile_selection("name CA", &mut system).unwrap();
        assert_eq!(sel.indices.as_slice(), &[1]);
    }

    #[test]
    fn selection_resid_range() {
        let mut system = build_system();
        let sel = compile_selection("resid 1:2", &mut system).unwrap();
        assert_eq!(sel.indices.len(), 4);
    }

    #[test]
    fn selection_protein_backbone() {
        let mut system = build_system();
        let sel = compile_selection("protein and backbone", &mut system).unwrap();
        assert_eq!(sel.indices.as_slice(), &[0, 1]);
    }

    #[test]
    fn selection_chain() {
        let mut system = build_system();
        let sel = compile_selection("chain A", &mut system).unwrap();
        assert_eq!(sel.indices.as_slice(), &[0, 1, 2]);
    }
}
