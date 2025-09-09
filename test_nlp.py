from app.nlp.text_models import classify_issue, sentiment_label

samples = [
    "The headphones arrived with a cracked earcup.",
    "Package was delivered 5 days late, needed it for a gift.",
    "Laptop stand was missing from the box.",
    "Thanks, everything arrived on time and works perfectly!",
    "I didn't receive anything. This is frustrating.",
    "",
]

def main():
    for i, s in enumerate(samples, 1):
        issue = classify_issue(s)
        sent  = sentiment_label(s)
        print(f"{i}. text={s!r}\n   issue={issue}  sentiment={sent}")

if __name__ == "__main__":
    main()
