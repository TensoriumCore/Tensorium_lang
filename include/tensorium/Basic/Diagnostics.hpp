#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

#if !defined(_WIN32)
#include <unistd.h>
#endif

namespace tensorium {

enum class DiagnosticLevel { Error, Warning, Note };

enum class ColorMode { Auto, Always, Never };

struct SourceLocation {
  int line = 0;
  int column = 0;
  int length = 1;

  bool isValid() const { return line > 0 && column > 0; }
};

class DiagnosticError final : public std::runtime_error {
  DiagnosticLevel level_;
  std::string message_;
  SourceLocation location_;
  std::string code_;

public:
  DiagnosticError(DiagnosticLevel level, std::string message,
                  SourceLocation location = {},
                  std::string code = "")
      : std::runtime_error(message), level_(level), message_(std::move(message)),
        location_(location), code_(std::move(code)) {}

  DiagnosticLevel level() const { return level_; }
  const std::string &message() const { return message_; }
  const SourceLocation &location() const { return location_; }
  const std::string &code() const { return code_; }
};

struct PrintDiagnosticOptions {
  ColorMode colorMode = ColorMode::Auto;
  bool showSourceLine = true;
};

inline const char *diagnosticLevelName(DiagnosticLevel level) {
  switch (level) {
  case DiagnosticLevel::Error:
    return "error";
  case DiagnosticLevel::Warning:
    return "warning";
  case DiagnosticLevel::Note:
    return "note";
  }
  return "error";
}

inline const char *diagnosticLevelColor(DiagnosticLevel level) {
  switch (level) {
  case DiagnosticLevel::Error:
    return "\033[31m";
  case DiagnosticLevel::Warning:
    return "\033[33m";
  case DiagnosticLevel::Note:
    return "\033[36m";
  }
  return "\033[31m";
}

inline bool stderrSupportsColor() {
  if (std::getenv("NO_COLOR"))
    return false;
  const char *term = std::getenv("TERM");
  if (!term || std::string_view(term) == "dumb")
    return false;
#if defined(_WIN32)
  return false;
#else
  return ::isatty(fileno(stderr)) != 0;
#endif
}

inline bool shouldUseColor(ColorMode mode) {
  switch (mode) {
  case ColorMode::Always:
    return true;
  case ColorMode::Never:
    return false;
  case ColorMode::Auto:
    return stderrSupportsColor();
  }
  return false;
}

inline std::string_view getSourceLine(std::string_view source, int oneBasedLine) {
  if (oneBasedLine <= 0)
    return {};
  size_t start = 0;
  int currentLine = 1;
  while (currentLine < oneBasedLine) {
    const size_t nl = source.find('\n', start);
    if (nl == std::string_view::npos)
      return {};
    start = nl + 1;
    ++currentLine;
  }
  size_t end = source.find('\n', start);
  if (end == std::string_view::npos)
    end = source.size();
  if (end > start && source[end - 1] == '\r')
    --end;
  return source.substr(start, end - start);
}

inline std::string expandTabs(std::string_view line, int tabWidth = 4) {
  std::string out;
  out.reserve(line.size());
  int visualCol = 0;
  for (char c : line) {
    if (c == '\t') {
      const int n = tabWidth - (visualCol % tabWidth);
      out.append(static_cast<size_t>(n), ' ');
      visualCol += n;
      continue;
    }
    out.push_back(c);
    ++visualCol;
  }
  return out;
}

inline size_t displayColumnOffset(std::string_view line, int oneBasedColumn,
                                  int tabWidth = 4) {
  if (oneBasedColumn <= 1)
    return 0;
  size_t offset = 0;
  int logicalCol = 1;
  for (char c : line) {
    if (logicalCol >= oneBasedColumn)
      break;
    if (c == '\t') {
      const int n = tabWidth - static_cast<int>(offset % tabWidth);
      offset += static_cast<size_t>(n);
    } else {
      ++offset;
    }
    ++logicalCol;
  }
  return offset;
}

inline void printDiagnostic(std::ostream &os, const std::string &filePath,
                            std::string_view source, DiagnosticLevel level,
                            const std::string &message,
                            SourceLocation location = {},
                            const std::string &code = "",
                            const PrintDiagnosticOptions &options = {}) {
  const bool useColor = shouldUseColor(options.colorMode);
  const char *bold = useColor ? "\033[1m" : "";
  const char *reset = useColor ? "\033[0m" : "";
  const char *levelColor = useColor ? diagnosticLevelColor(level) : "";

  const std::string displayFile = filePath.empty() ? "<input>" : filePath;
  os << displayFile;
  if (location.isValid()) {
    os << ":" << location.line << ":" << std::max(1, location.column);
  }
  os << ": " << levelColor << bold << diagnosticLevelName(level) << reset
     << ": " << message;
  if (!code.empty())
    os << " [" << code << "]";
  os << "\n";

  if (!options.showSourceLine || !location.isValid() || source.empty())
    return;

  const std::string_view rawLine = getSourceLine(source, location.line);
  if (rawLine.empty())
    return;

  const std::string displayLine = expandTabs(rawLine);
  const size_t digits = std::to_string(std::max(1, location.line)).size();
  os << std::setw(static_cast<int>(digits)) << location.line << " | "
     << displayLine << "\n";

  const size_t caretPad = displayColumnOffset(rawLine, location.column);
  const int caretLength = std::max(1, location.length);
  os << std::string(digits, ' ') << " | " << std::string(caretPad, ' ')
     << levelColor << bold << "^";
  for (int i = 1; i < caretLength; ++i)
    os << "~";
  os << reset << "\n";
}

} // namespace tensorium
