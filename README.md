# Promptly 🌐🔒

**Promptly** is a minimalist, secure web browser with Vim-like functionality and advanced productivity features. It is designed for users who value efficiency, configurability, and privacy. 

## Features

### Core Features:
- **Minimalist Browser**:
  - Lightweight design focused on performance and simplicity.
  - Embedded ChatGPT extension for seamless interaction.
- **Vim-Like Shortcuts**:
  - Use commands like `:vsplit`, `:split`, `:!gh`, and more for quick navigation and multitasking.
  - Fully configurable via YAML files.
- **Tiling Support**:
  - Vertical (`:vsplit`) and horizontal (`:split`) splits for a tiling-like experience.
- **Search Engine Customization**:
  - Configurable search shortcuts (e.g., `:!yt` for YouTube, `:!reddit` for Reddit).

### Optional Enhancements:
- **Tab Management**:
  - Commands like `:tabnew` and `:tabclose` for organizing multiple tabs.
- **Ad-Blocking and Tracker Blocking**:
  - Integrated ad-blocker for a distraction-free browsing experience.
- **Bookmark and History Management**:
  - Save and search bookmarks and browsing history.
- **Offline Mode**:
  - Cache pages for offline reading.

## 🛠️ Tech Stack

Promptly is built with modern, high-performance tools to ensure reliability and extensibility:

### **Core Technology**
- **[Rust](https://www.rust-lang.org/)**:
  - Chosen for its memory safety, speed, and concurrency model, Rust ensures a secure and efficient browsing experience.
- **[WebKitGTK](https://webkitgtk.org/)**:
  - A lightweight, open-source rendering engine that supports modern web standards (HTML5, CSS3, JavaScript).
- **[GTK](https://www.gtk.org/)**:
  - A versatile, cross-platform UI library for building clean, user-friendly interfaces.

### **Key Libraries**
- **[serde](https://serde.rs/)** and **[serde_yaml](https://docs.rs/serde_yaml)**:
  - For YAML-based configuration management.
- **[rustls](https://docs.rs/rustls/)**:
  - Ensures secure communication with HTTPS enforcement.
- **[easylist_parser](https://github.com/twmb/easylist-rust)**:
  - Parses blocklists to support ad-blocking.
- **[gtk-rs](https://gtk-rs.org/)**:
  - Provides bindings to GTK and WebKitGTK for seamless integration with Rust.

### **Why Rust?**
- **Performance**: Rust delivers near-C/C++ speeds, ensuring fast rendering and low resource usage.
- **Memory Safety**: With Rust's ownership model, Promptly is free of common memory-related vulnerabilities.
- **Concurrency**: Rust's async model allows smooth multitasking (e.g., tabs, splits, and parallel queries).

---

### Focus Areas:
- **Security**:
  - Sandboxing for safe browsing.
  - HTTPS enforced by default.
- **Rendering Quality**:
  - Modern web standards support (HTML5, CSS3, JS).

---

## Installation

Promptly is not yet available for use. Development will begin in **2025**, and installation instructions will be provided upon release.

---

## 📜 Configuration

Promptly uses a YAML configuration file for customization. Example:
```yaml
search_engines:
  default: "duckduckgo"
  shortcuts:
    yt: "https://www.youtube.com/results?search_query="
    gh: "https://github.com/search?q="
    reddit: "https://www.reddit.com/search/?q="

keybindings:
  vsplit: "Ctrl+v"
  hsplit: "Ctrl+h"
  close_split: "Ctrl+w"
```

---

## **📅 Roadmap**

### Phase 1: Core Functionality
- Basic browser with YAML-configurable shortcuts.
- Integrated ChatGPT extension.
- Vertical and horizontal splits

### Phase 2: Optional Enhancements
- Tab management, ad-blocking, and bookmarks.
- Offline mode and user-defined scripts.

### Phase 3: Security and Optimization
- Implement sandboxing and HTTPS enforcement.
- Optimize performance for rendering and multitasking.

---

## 📝 License

This project is licensed under the **GNU General Public License v3.0**. You are free to use, modify, and distribute this software under the terms of the GPL. Any derivative works must also be licensed under the GPL.

See the [LICENSE](LICENSE) file for details.

---

## **Contact**
- **Author**: [Aftaab Siddiqui](https://github.com/MaskedSyntax)
- **Repository**: [Promptly](https://github.com/MaskedSyntax/Promptly)

Feel free to reach out with ideas, feedback, or questions!
