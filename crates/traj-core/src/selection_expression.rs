use crate::error::{TrajError, TrajResult};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelectionExpr {
    All,
    Predicate(SelectionPredicate),
    Not(Box<SelectionExpr>),
    And(Box<SelectionExpr>, Box<SelectionExpr>),
    Or(Box<SelectionExpr>, Box<SelectionExpr>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelectionPredicate {
    Name(Vec<String>),
    Resname(Vec<String>),
    Resid(Vec<ResidRange>),
    Chain(Vec<String>),
    Element(Vec<String>),
    Protein,
    Backbone,
    SideChain,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidRange {
    pub start: i32,
    pub end: i32,
}

impl ResidRange {
    pub fn contains(&self, resid: i32) -> bool {
        let (lo, hi) = if self.start <= self.end {
            (self.start, self.end)
        } else {
            (self.end, self.start)
        };
        resid >= lo && resid <= hi
    }
}

pub const PROTEIN_RESNAMES: &[&str] = &[
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET",
    "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "MSE", "HSD", "HSE", "HSP",
];

pub const BACKBONE_ATOM_NAMES: &[&str] = &["N", "CA", "C", "O", "OXT"];

pub fn is_protein_resname(name: &str) -> bool {
    let upper = name.to_ascii_uppercase();
    PROTEIN_RESNAMES.contains(&upper.as_str())
}

pub fn is_backbone_atom(name: &str) -> bool {
    let upper = name.to_ascii_uppercase();
    BACKBONE_ATOM_NAMES.contains(&upper.as_str())
}

pub fn is_sidechain_heavy_atom(atom_name: &str, element: &str) -> bool {
    !is_backbone_atom(atom_name) && !element.eq_ignore_ascii_case("H")
}

pub fn parse_selection_expression(expr: &str) -> TrajResult<SelectionExpr> {
    let tokens = tokenize(expr);
    if tokens.is_empty() {
        return Ok(SelectionExpr::All);
    }
    let mut parser = Parser::new(tokens);
    let expr = parser.parse_expr()?;
    if parser.pos < parser.tokens.len() {
        return Err(TrajError::InvalidSelection(format!(
            "unexpected token '{}'",
            parser.tokens[parser.pos]
        )));
    }
    Ok(expr)
}

fn tokenize(input: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut chars = input.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch.is_whitespace() {
            continue;
        }
        if matches!(ch, '(' | ')' | ':') {
            tokens.push(ch.to_string());
            continue;
        }
        let mut token = String::from(ch);
        while let Some(&next) = chars.peek() {
            if next.is_whitespace() || matches!(next, '(' | ')' | ':') {
                break;
            }
            token.push(next);
            chars.next();
        }
        tokens.push(token);
    }
    tokens
}

fn is_boundary_token(token: &str) -> bool {
    matches!(token, "(" | ")" | ":")
        || matches!(
            token.to_ascii_lowercase().as_str(),
            "and"
                | "or"
                | "not"
                | "all"
                | "name"
                | "resname"
                | "resid"
                | "chain"
                | "element"
                | "protein"
                | "backbone"
                | "sidechain"
        )
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
        self.tokens.get(self.pos).map(|token| token.as_str())
    }

    fn next(&mut self) -> Option<String> {
        if self.pos < self.tokens.len() {
            let token = self.tokens[self.pos].clone();
            self.pos += 1;
            Some(token)
        } else {
            None
        }
    }

    fn parse_expr(&mut self) -> TrajResult<SelectionExpr> {
        self.parse_or()
    }

    fn parse_or(&mut self) -> TrajResult<SelectionExpr> {
        let mut left = self.parse_and()?;
        while self
            .peek()
            .map(|token| token.eq_ignore_ascii_case("or"))
            .unwrap_or(false)
        {
            self.next();
            let right = self.parse_and()?;
            left = SelectionExpr::Or(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> TrajResult<SelectionExpr> {
        let mut left = self.parse_not()?;
        while self
            .peek()
            .map(|token| token.eq_ignore_ascii_case("and"))
            .unwrap_or(false)
        {
            self.next();
            let right = self.parse_not()?;
            left = SelectionExpr::And(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_not(&mut self) -> TrajResult<SelectionExpr> {
        if self
            .peek()
            .map(|token| token.eq_ignore_ascii_case("not"))
            .unwrap_or(false)
        {
            self.next();
            let inner = self.parse_not()?;
            Ok(SelectionExpr::Not(Box::new(inner)))
        } else {
            self.parse_primary()
        }
    }

    fn parse_primary(&mut self) -> TrajResult<SelectionExpr> {
        let token = self
            .peek()
            .ok_or_else(|| TrajError::InvalidSelection("unexpected end of selection".into()))?
            .to_string();
        if token == "(" {
            self.next();
            let inner = self.parse_expr()?;
            match self.next().as_deref() {
                Some(")") => return Ok(inner),
                _ => {
                    return Err(TrajError::InvalidSelection("expected ')'".into()));
                }
            }
        }

        let keyword = token.to_ascii_lowercase();
        let predicate = match keyword.as_str() {
            "all" => {
                self.next();
                return Ok(SelectionExpr::All);
            }
            "protein" => {
                self.next();
                SelectionPredicate::Protein
            }
            "backbone" => {
                self.next();
                SelectionPredicate::Backbone
            }
            "sidechain" => {
                self.next();
                SelectionPredicate::SideChain
            }
            "name" => {
                self.next();
                SelectionPredicate::Name(self.parse_values()?)
            }
            "resname" => {
                self.next();
                SelectionPredicate::Resname(self.parse_values()?)
            }
            "resid" => {
                self.next();
                SelectionPredicate::Resid(self.parse_resid_values()?)
            }
            "chain" => {
                self.next();
                SelectionPredicate::Chain(self.parse_values()?)
            }
            "element" => {
                self.next();
                SelectionPredicate::Element(self.parse_values()?)
            }
            _ => {
                return Err(TrajError::InvalidSelection(format!(
                    "unexpected token '{token}'",
                )));
            }
        };
        Ok(SelectionExpr::Predicate(predicate))
    }

    fn parse_values(&mut self) -> TrajResult<Vec<String>> {
        let mut values = Vec::new();
        while let Some(token) = self.peek() {
            if is_boundary_token(token) {
                break;
            }
            values.push(self.next().unwrap().to_ascii_uppercase());
        }
        if values.is_empty() {
            return Err(TrajError::InvalidSelection(
                "expected value after keyword".into(),
            ));
        }
        Ok(values)
    }

    fn parse_resid_values(&mut self) -> TrajResult<Vec<ResidRange>> {
        let mut ranges = Vec::new();
        while let Some(token) = self.peek() {
            if is_boundary_token(token) {
                break;
            }
            let token = self.next().unwrap();
            if let Some(range) = parse_inline_resid_range(&token)? {
                ranges.push(range);
                continue;
            }
            let start = parse_resid_int(&token)?;
            if self.peek() == Some(":") {
                self.next();
                let end_token = self.next().ok_or_else(|| {
                    TrajError::InvalidSelection("expected resid after ':'".into())
                })?;
                if is_boundary_token(&end_token) {
                    return Err(TrajError::InvalidSelection(
                        "expected resid after ':'".into(),
                    ));
                }
                let end = parse_resid_int(&end_token)?;
                ranges.push(ResidRange { start, end });
            } else {
                ranges.push(ResidRange { start, end: start });
            }
        }
        if ranges.is_empty() {
            return Err(TrajError::InvalidSelection("expected resid value".into()));
        }
        Ok(ranges)
    }
}

fn parse_inline_resid_range(token: &str) -> TrajResult<Option<ResidRange>> {
    let split_at = token
        .char_indices()
        .skip(1)
        .find(|(_, ch)| matches!(ch, '-' | ':'))
        .map(|(idx, _)| idx);
    let Some(split_at) = split_at else {
        return Ok(None);
    };
    let start = parse_resid_int(&token[..split_at])?;
    let end = parse_resid_int(&token[split_at + 1..])?;
    Ok(Some(ResidRange { start, end }))
}

fn parse_resid_int(token: &str) -> TrajResult<i32> {
    token
        .parse()
        .map_err(|_| TrajError::InvalidSelection(format!("invalid residue id '{token}'")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_empty_as_all() {
        assert_eq!(parse_selection_expression("").unwrap(), SelectionExpr::All);
    }

    #[test]
    fn parses_multi_value_name_expression() {
        let expr = parse_selection_expression("name CA CB and not element H").unwrap();
        assert!(matches!(expr, SelectionExpr::And(_, _)));
    }

    #[test]
    fn parses_residue_ranges() {
        let expr = parse_selection_expression("resid 1-3 8 : 10").unwrap();
        let SelectionExpr::Predicate(SelectionPredicate::Resid(ranges)) = expr else {
            panic!("unexpected expression");
        };
        assert_eq!(
            ranges,
            vec![
                ResidRange { start: 1, end: 3 },
                ResidRange { start: 8, end: 10 },
            ]
        );
    }

    #[test]
    fn rejects_unknown_predicate() {
        assert!(parse_selection_expression("bogus").is_err());
    }
}
