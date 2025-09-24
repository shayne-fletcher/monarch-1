//! This module contains a lexer for Hyperactor identifiers.

use std::iter::Peekable;
use std::str::Chars;

/// An identifier token.
#[derive(Debug, PartialEq, Eq)]
pub enum Token {
    /// "["
    LeftBracket,
    /// "]"
    RightBracket,
    /// A decimal unsigned integer.
    Uint(usize),
    /// "."
    Dot,
    /// "//"
    DoubleSlash,
    /// Colon, within a bracket
    BracketColon,
    /// Typename, within a bracket. These are rust identifiers, plus
    /// ':' characters.
    BracketTypename(String),
    /// An identifier, following rust rules, and allowing for '-'
    /// as a continuation character.
    Ident(String),

    /// Special token to denote a lexer error. It contains the unlexed
    /// remainder of the input. The error occured on the first character.
    Error(String),
}

/// A lexer is an iterator over [`Token`].
#[derive(Default)]
enum Lexer<'a> {
    Next(Peekable<Chars<'a>>),
    Inbracket(Peekable<Chars<'a>>),
    #[default]
    Invalid,
}

impl<'a> Lexer<'a> {
    /// Create a new lexer over the provided input.
    fn new(input: &'a str) -> Self {
        Lexer::Next(input.chars().peekable())
    }

    fn take_uint(iter: &mut Peekable<Chars<'a>>) -> Token {
        let mut value = iter.next().unwrap().to_digit(10).unwrap() as usize;
        while let Some(&ch) = iter.peek() {
            if let Some(d) = ch.to_digit(10) {
                value = value * 10 + d as usize;
                iter.next();
            } else {
                break;
            }
        }
        Token::Uint(value)
    }

    fn take_ident(iter: &mut Peekable<Chars<'a>>) -> Token {
        let mut ident = String::new();

        let ch = iter.next().unwrap();
        assert!(unicode_ident::is_xid_start(ch) || ch == '_');
        ident.push(ch);

        while let Some(&ch) = iter.peek()
            && (unicode_ident::is_xid_continue(ch) || ch == '-')
        {
            ident.push(iter.next().unwrap());
        }
        Token::Ident(ident)
    }

    fn take_bracket_typename(iter: &mut Peekable<Chars<'a>>) -> Token {
        let mut ident = String::new();

        let ch = iter.next().unwrap();
        assert!(unicode_ident::is_xid_start(ch) || ch == '_');
        ident.push(ch);

        while let Some(&ch) = iter.peek()
            && (unicode_ident::is_xid_continue(ch) || ch == ':')
        {
            ident.push(iter.next().unwrap());
        }
        Token::BracketTypename(ident)
    }
}

impl<'a> Iterator for Lexer<'a> {
    type Item = Token;

    fn next(&mut self) -> Option<Self::Item> {
        let (new_state, token) = match std::mem::take(self) {
            Lexer::Next(mut iter) => match iter.peek()? {
                '[' => {
                    let _ = iter.next();
                    (Lexer::Inbracket(iter), Some(Token::LeftBracket))
                }
                '.' => {
                    let _ = iter.next();
                    (Lexer::Next(iter), Some(Token::Dot))
                }
                '/' => {
                    let _ = iter.next();
                    match iter.next() {
                        Some('/') => (Lexer::Next(iter), Some(Token::DoubleSlash)),
                        _ => (Lexer::Invalid, Some(Token::Error(iter.collect()))),
                    }
                }
                // TODO: support hexadecimal
                '0'..='9' => {
                    let token = Lexer::take_uint(&mut iter);
                    (Lexer::Next(iter), Some(token))
                }
                ch if unicode_ident::is_xid_start(*ch) || *ch == '_' => {
                    let token = Lexer::take_ident(&mut iter);
                    (Lexer::Next(iter), Some(token))
                }
                _ => (Lexer::Invalid, Some(Token::Error(iter.collect()))),
            },
            Lexer::Inbracket(mut iter) => match iter.peek()? {
                '0'..='9' => {
                    let token = Lexer::take_uint(&mut iter);
                    (Lexer::Inbracket(iter), Some(token))
                }
                ch if unicode_ident::is_xid_start(*ch) => {
                    let token = Lexer::take_bracket_typename(&mut iter);
                    (Lexer::Inbracket(iter), Some(token))
                }
                ':' => {
                    let _ = iter.next();
                    (Lexer::Inbracket(iter), Some(Token::BracketColon))
                }
                ']' => {
                    let _ = iter.next();
                    (Lexer::Next(iter), Some(Token::RightBracket))
                }
                _ => (Lexer::Invalid, Some(Token::Error(iter.collect()))),
            },
            Lexer::Invalid => (Lexer::Invalid, None),
        };

        *self = new_state;
        token
    }
}

#[cfg(test)]
mod tests {
    use Token::*;

    use super::*;

    #[test]
    fn test_basic() {
        assert_eq!(
            Lexer::new("foo.bar[123:foo]//blah").collect::<Vec<_>>(),
            vec![
                Ident("foo".to_string()),
                Dot,
                Ident("bar".to_string()),
                LeftBracket,
                Uint(123),
                BracketColon,
                BracketTypename("foo".to_string()),
                RightBracket,
                DoubleSlash,
                Ident("blah".to_string())
            ]
        );
        assert_eq!(
            Lexer::new("foo.ba)r[123:foo]//blah").collect::<Vec<_>>(),
            vec![
                Ident("foo".to_string()),
                Dot,
                Ident("ba".to_string()),
                Error(")r[123:foo]//blah".to_string())
            ]
        );
    }
}
