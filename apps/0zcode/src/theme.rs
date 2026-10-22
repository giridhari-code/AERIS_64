//! Theme: "blood and ash". All widget colors come from here.

use ratatui::style::{Color, Modifier, Style};
use std::env;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorMode {
    True,
    Ansi16,
    None,
}

#[derive(Debug, Clone)]
pub struct Theme {
    pub mode: ColorMode,
    pub light: bool,
    pub bg: Color,
    pub panel: Color,
    pub border: Color,
    pub text: Color,
    pub text_dim: Color,
    pub text_muted: Color,
    pub accent: Color,
    pub danger: Color,
    pub success: Color,
    pub warning: Color,
}

impl Theme {
    pub fn blood_and_ash() -> Self {
        Self {
            mode: ColorMode::True,
            light: false,
            bg: Color::Rgb(0x0b, 0x0b, 0x0c),
            panel: Color::Rgb(0x16, 0x16, 0x1a),
            border: Color::Rgb(0x2a, 0x2a, 0x2e),
            text: Color::Rgb(0xe8, 0xe6, 0xe3),
            text_dim: Color::Rgb(0x8e, 0x8e, 0x93),
            text_muted: Color::Rgb(0x5e, 0x5e, 0x64),
            accent: Color::Rgb(0xc8, 0x10, 0x2e),
            danger: Color::Rgb(0xf0, 0x52, 0x5b),
            success: Color::Rgb(0x3f, 0xb8, 0xaf),
            warning: Color::Rgb(0xc9, 0x8a, 0x3d),
        }
    }

    pub fn light() -> Self {
        Self {
            mode: ColorMode::True,
            light: true,
            bg: Color::Rgb(0xf5, 0xf4, 0xf2),
            panel: Color::Rgb(0xff, 0xff, 0xff),
            border: Color::Rgb(0xd0, 0xce, 0xca),
            text: Color::Rgb(0x1c, 0x1c, 0x1e),
            text_dim: Color::Rgb(0x6b, 0x6b, 0x70),
            text_muted: Color::Rgb(0x9a, 0x9a, 0xa0),
            accent: Color::Rgb(0xa0, 0x0c, 0x24),
            danger: Color::Rgb(0xc0, 0x2e, 0x36),
            success: Color::Rgb(0x1f, 0x8a, 0x82),
            warning: Color::Rgb(0x9a, 0x6a, 0x28),
        }
    }

    pub fn ansi16() -> Self {
        let mut t = Self::blood_and_ash();
        t.mode = ColorMode::Ansi16;
        t.bg = Color::Black;
        t.panel = Color::DarkGray;
        t.border = Color::Gray;
        t.text = Color::White;
        t.text_dim = Color::Gray;
        t.text_muted = Color::DarkGray;
        t.accent = Color::Red;
        t.danger = Color::LightRed;
        t.success = Color::Cyan;
        t.warning = Color::Yellow;
        t
    }

    pub fn none() -> Self {
        let mut t = Self::blood_and_ash();
        t.mode = ColorMode::None;
        t.bg = Color::Reset;
        t.panel = Color::Reset;
        t.border = Color::Reset;
        t.text = Color::Reset;
        t.text_dim = Color::Reset;
        t.text_muted = Color::Reset;
        t.accent = Color::Reset;
        t.danger = Color::Reset;
        t.success = Color::Reset;
        t.warning = Color::Reset;
        t
    }

    pub fn from_env() -> Self {
        if env::var_os("NO_COLOR").is_some()
            || env::var("ZEROZ_NO_COLOR").ok().as_deref() == Some("1")
            || env::var("TERM").ok().as_deref() == Some("dumb")
        {
            return Self::none();
        }
        if env::var("ZEROZ_COLOR").ok().as_deref() == Some("16") {
            return Self::ansi16();
        }
        if env::var("ZEROZ_THEME").ok().as_deref() == Some("light") {
            return Self::light();
        }
        Self::blood_and_ash()
    }

    pub fn style_text(&self) -> Style {
        Style::default().fg(self.text)
    }
    pub fn style_dim(&self) -> Style {
        Style::default().fg(self.text_dim)
    }
    pub fn style_muted(&self) -> Style {
        Style::default().fg(self.text_muted)
    }
    pub fn style_accent(&self) -> Style {
        Style::default().fg(self.accent).add_modifier(Modifier::BOLD)
    }
    pub fn style_danger(&self) -> Style {
        Style::default().fg(self.danger)
    }
    pub fn style_success(&self) -> Style {
        Style::default().fg(self.success)
    }
    pub fn style_warning(&self) -> Style {
        Style::default().fg(self.warning)
    }
    pub fn style_border(&self) -> Style {
        Style::default().fg(self.border)
    }
    pub fn style_panel(&self) -> Style {
        Style::default().bg(self.panel).fg(self.text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accent_is_blood_red() {
        let t = Theme::blood_and_ash();
        assert_eq!(t.accent, Color::Rgb(0xc8, 0x10, 0x2e));
    }

    #[test]
    fn success_is_teal_not_green() {
        let t = Theme::blood_and_ash();
        assert_eq!(t.success, Color::Rgb(0x3f, 0xb8, 0xaf));
    }
}
