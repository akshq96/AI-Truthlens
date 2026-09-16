import React, { useState } from 'react';
import { Link } from 'react-router-dom';

const FAQ_ITEMS = [
  { q: 'What image formats are supported?', a: 'We support JPEG, PNG, and WEBP images up to 5MB.' },
  { q: 'How accurate is the AI detection?', a: 'Our analysis combines model patterns, artifact detection, and metadata. Results are probability scores, not guarantees.' },
  { q: 'Is my uploaded image stored?', a: 'Images are processed for analysis. We do not store or share your uploads beyond what\'s needed to return results.' },
  { q: 'What does the score breakdown mean?', a: 'Model score reflects AI-like patterns, artifact score checks for visual inconsistencies, and metadata score evaluates EXIF data.' },
  { q: 'Why might metadata be missing?', a: 'AI-generated images often lack camera EXIF data. Real photos typically include camera make, model, and timestamp.' },
];

export default function ContactPage() {
  const [form, setForm] = useState({ firstName: '', lastName: '', email: '', message: '' });
  const handleChange = e => setForm(p => ({ ...p, [e.target.name]: e.target.value }));

  return (
    <div className="contact-page">
      <header className="contact-page__header">
        <div className="contact-page__kicker">✉️ Contact us</div>
        <h1 className="contact-page__title">Get in touch</h1>
      </header>

      <div className="contact-page__grid">
        {/* Form panel */}
        <div className="contact-glass">
          <h2 className="contact__title">Send a message</h2>
          <form className="contact__form" onSubmit={e => e.preventDefault()}>
            <div className="contact__row">
              <div className="contact__field">
                <label htmlFor="firstName">First name</label>
                <input id="firstName" name="firstName" type="text" value={form.firstName} onChange={handleChange} placeholder="Jane" />
              </div>
              <div className="contact__field">
                <label htmlFor="lastName">Last name</label>
                <input id="lastName" name="lastName" type="text" value={form.lastName} onChange={handleChange} placeholder="Doe" />
              </div>
            </div>
            <div className="contact__field">
              <label htmlFor="email">Email</label>
              <input id="email" name="email" type="email" value={form.email} onChange={handleChange} placeholder="you@example.com" />
            </div>
            <div className="contact__field">
              <label htmlFor="message">Message</label>
              <textarea id="message" name="message" value={form.message} onChange={handleChange} placeholder="Tell us what's on your mind..." rows={5} />
            </div>
            <button type="submit" className="contact__submit">
              <span className="contact__submit-dot" aria-hidden="true" />
              Send message
            </button>
          </form>

          <div className="contact__email">
            <span className="contact__email-label">Or email us directly</span>
            <a href="mailto:info@aideepfake.com" className="contact__email-link">info@aideepfake.com</a>
          </div>
        </div>

        {/* FAQ panel */}
        <div className="contact-glass">
          <h2 className="contact__faq-title">Frequently asked questions</h2>
          <div className="contact__faq-list">
            {FAQ_ITEMS.map((item, i) => (
              <details key={i} className="contact__faq-item">
                <summary className="contact__faq-question">{item.q}</summary>
                <p className="contact__faq-answer">{item.a}</p>
              </details>
            ))}
          </div>
        </div>
      </div>

      <div className="contact__back">
        <Link to="/" className="contact__back-link">← Back to home</Link>
      </div>
    </div>
  );
}
