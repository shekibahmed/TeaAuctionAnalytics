# Contributing to CTC Tea Sales Analytics Dashboard

Thank you for your interest in contributing to the CTC Tea Sales Analytics Dashboard! This document provides guidelines for contributing to the project.

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Git for version control
- Basic understanding of Streamlit, Pandas, and data visualization

### Development Setup

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/tea-sales-analytics.git
   cd tea-sales-analytics
   ```

3. **Install dependencies**:
   ```bash
   pip install -e .
   ```

4. **Set up environment variables** (optional):
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   ```

5. **Run the application** locally:
   ```bash
   streamlit run main.py --server.port=5000 --server.address=0.0.0.0
   ```

## How to Contribute

### Reporting Issues

Before creating a new issue, please:

1. **Search existing issues** to avoid duplicates
2. **Check the troubleshooting section** in README.md
3. **Provide detailed information**:
   - Operating system and Python version
   - Streamlit version
   - Steps to reproduce the issue
   - Expected vs actual behavior
   - Sample data (if applicable)
   - Error messages or screenshots

### Suggesting Features

When suggesting new features:

1. **Check existing feature requests** first
2. **Describe the problem** the feature would solve
3. **Explain the proposed solution** in detail
4. **Consider the impact** on existing functionality
5. **Provide use cases** and examples

### Code Contributions

#### Types of Contributions Welcome

- **Bug fixes**: Fix existing functionality issues
- **Feature enhancements**: Add new analysis capabilities
- **Data processing improvements**: Better file handling and validation
- **UI/UX improvements**: Enhanced user interface and experience
- **Performance optimizations**: Faster data processing and visualization
- **Documentation**: Improve or expand documentation
- **Testing**: Add or improve test coverage
- **Animation enhancements**: New tea-themed loading animations

#### Development Guidelines

1. **Code Style**:
   - Follow PEP 8 style guidelines
   - Use meaningful variable and function names
   - Add docstrings for functions and classes
   - Keep functions focused and single-purpose

2. **File Organization**:
   - `main.py`: Main Streamlit application
   - `utils.py`: Data processing and analysis functions
   - `styles.py`: CSS styling and UI components
   - `loading_animations.py`: Animation system
   - `assets/`: Sample data and static resources

3. **Tea Theme Consistency**:
   - Maintain tea-related terminology in animations
   - Use appropriate tea metaphors for processing steps
   - Keep the warm, welcoming tone throughout the UI

#### Making Changes

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Write clean, well-documented code
   - Test your changes thoroughly
   - Ensure the application runs without errors

3. **Test your changes**:
   - Upload different file formats
   - Test with various data sizes
   - Verify all tabs and features work
   - Check mobile responsiveness

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Add feature: descriptive commit message"
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request**:
   - Use a clear, descriptive title
   - Explain what your changes do
   - Reference any related issues
   - Include screenshots for UI changes

#### Pull Request Guidelines

**Before submitting**:
- [ ] Code follows the project style guidelines
- [ ] All existing functionality continues to work
- [ ] New features include appropriate documentation
- [ ] Changes are tested with sample data
- [ ] Commit messages are clear and descriptive

**Pull Request Description**:
- **What**: Brief description of changes
- **Why**: Reason for the changes
- **How**: How the changes work
- **Testing**: How you tested the changes
- **Screenshots**: For UI/visual changes

## Code Areas

### Core Application (`main.py`)
- Main Streamlit interface
- Tab organization and navigation
- File upload and processing
- Chart rendering and layout

### Data Processing (`utils.py`)
- Excel/CSV file parsing
- Data validation and cleaning
- Statistical calculations
- AI integration functions

### Styling (`styles.py`)
- Custom CSS styles
- Mobile responsiveness
- Theme customization

### Animations (`loading_animations.py`)
- Tea-themed loading indicators
- Progress tracking
- Animation timing and effects

## Development Tips

### Working with Streamlit
- Use `st.rerun()` for dynamic updates
- Cache expensive operations with `@st.cache_data`
- Test with different screen sizes
- Handle user input validation gracefully

### Data Handling Best Practices
- Validate file uploads before processing
- Handle missing or malformed data gracefully
- Optimize for large datasets
- Provide clear error messages

### Animation Development
- Keep animations lightweight and performant
- Use appropriate tea metaphors
- Ensure animations don't block data processing
- Test animation timing on different devices

## Testing Guidelines

### Manual Testing Checklist
- [ ] File upload works with Excel and CSV files
- [ ] All market selection filters function properly
- [ ] Charts render correctly in all tabs
- [ ] AI features work (if API key provided)
- [ ] Mobile interface is responsive
- [ ] Animations display properly
- [ ] Error handling works for invalid files

### Sample Data Testing
- Use the provided `assets/default_data.csv`
- Test with different market combinations
- Verify calculations with known data sets
- Test edge cases (empty files, single rows, etc.)

## Documentation

When contributing documentation:
- Use clear, concise language
- Include code examples where helpful
- Update README.md for new features
- Add inline comments for complex logic
- Update this CONTRIBUTING.md if process changes

## Community Guidelines

### Code of Conduct
- Be respectful and inclusive
- Help newcomers get started
- Provide constructive feedback
- Focus on the project goals

### Communication
- Use issue comments for discussion
- Ask questions if unclear about requirements
- Provide context for your contributions
- Be patient with review processes

## Release Process

Maintainers handle releases, but contributors should:
- Update version numbers in pull requests (if applicable)
- Note breaking changes in pull request descriptions
- Update documentation for new features
- Consider backward compatibility

## Getting Help

If you need help:
1. Check the README.md troubleshooting section
2. Look through existing issues and discussions
3. Create a new issue with the "question" label
4. Provide detailed context about what you're trying to achieve

## Recognition

Contributors will be recognized in:
- GitHub contributors list
- Release notes for significant contributions
- README.md acknowledgments section

---

Thank you for contributing to the CTC Tea Sales Analytics Dashboard! Your efforts help make tea market analysis more accessible and insightful for everyone.