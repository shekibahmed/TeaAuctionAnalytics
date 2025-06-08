# Security Policy

## Supported Versions

We release patches for security vulnerabilities in the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in the CTC Tea Sales Analytics Dashboard, please report it responsibly.

### How to Report

1. **Email**: Send details to the repository maintainer
2. **GitHub Security Advisory**: Use GitHub's private vulnerability reporting feature
3. **Do NOT** create public issues for security vulnerabilities

### What to Include

When reporting a security issue, please include:

- Description of the vulnerability
- Steps to reproduce the issue
- Potential impact assessment
- Suggested fix (if available)
- Your contact information

### Response Timeline

- **Initial Response**: Within 48 hours
- **Assessment**: Within 7 days
- **Fix Development**: Depends on severity
- **Public Disclosure**: After fix is deployed

## Security Considerations

### Data Handling

- **File Uploads**: Only Excel (.xlsx, .xls) and CSV files are accepted
- **Data Processing**: All data remains client-side in the browser session
- **No Persistent Storage**: Uploaded data is not stored on servers
- **Memory Cleanup**: Data is cleared when session ends

### API Keys

- **OpenAI Integration**: API keys should be stored as environment variables
- **Key Security**: Never commit API keys to version control
- **Access Control**: API keys are only used for AI analysis features

### Dependencies

We regularly monitor dependencies for known vulnerabilities:

- **Streamlit**: Web framework security updates
- **Pandas**: Data processing library updates
- **OpenAI**: API client security patches
- **Plotly**: Visualization library updates

### Best Practices

#### For Users

- Keep your environment updated with latest package versions
- Use environment variables for sensitive configuration
- Validate data sources before uploading
- Monitor API usage for unusual patterns

#### For Developers

- Follow secure coding practices
- Validate all user inputs
- Sanitize file uploads
- Use parameterized queries if database integration is added
- Implement proper error handling without exposing sensitive information

### Known Security Features

- **Input Validation**: File format and size restrictions
- **Error Handling**: Graceful error messages without system information exposure
- **Session Management**: No persistent user sessions
- **HTTPS**: Recommended for production deployments

### Vulnerability Assessment

Regular security assessments include:

- Dependency vulnerability scanning
- Static code analysis
- Input validation testing
- API security review

## Security Updates

Security updates will be released as:

- **Critical**: Immediate patch release
- **High**: Patch within 7 days
- **Medium**: Next minor version
- **Low**: Next major version

Updates will be announced through:

- GitHub Security Advisories
- Release notes
- Repository notifications

## Compliance

This project follows security best practices for:

- Open source software development
- Data processing applications
- Web application security
- API integration security

## Contact

For security-related questions or concerns:

- Create a private security advisory on GitHub
- Contact repository maintainers directly
- Follow responsible disclosure practices

---

Thank you for helping keep the CTC Tea Sales Analytics Dashboard secure!