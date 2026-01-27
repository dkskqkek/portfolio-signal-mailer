# Skills Directory

**Welcome to the skills folder!** This is where all 230+ specialized AI skills live.

## 🤔 What Are Skills?

Skills are specialized instruction sets that teach AI assistants how to handle specific tasks. Think of them as expert knowledge modules that your AI can load on-demand.

**Simple analogy:** Just like you might consult different experts (a designer, a security expert, a marketer), skills let your AI become an expert in different areas when you need them.

---

## 📂 Folder Structure

The skills are now organized into categorized subfolders for easier navigation:

```
skills/
├── ai_agents/              # AI, Agents, LLMs, Prompts, RAG
├── security/               # Pentesting, Ethical Hacking, Security Audits
├── development/            # Languages, Frameworks, DBs, APIs
├── devops/                 # Cloud, Docker, Git, Linux, Deployment
├── product_design/         # UI/UX, Marketing, SEO, Copywriting
├── engineering/            # Architecture, Testing, Planning, Code Review
├── office/                 # Office Documents (Word, Excel, PPT, PDF)
│
└── [category]/[skill-name]/
    ├── SKILL.md            # Main skill definition
    ├── scripts/            # Helper scripts
    ├── examples/           # Usage examples
    └── resources/          # Templates & resources
```

**Key point:** To find a skill, check the relevant category folder.

---

## How to Use Skills

### Step 1: Make sure skills are installed
Skills should be in your `.agent/skills/` directory (or equivalent).

### Step 2: Invoke a skill in your AI chat
Use the `@` symbol followed by the skill name:

```
@brainstorming help me design a todo app
```

or

```
@stripe-integration add payment processing to my app
```

### Step 3: The AI becomes an expert
The AI loads that skill's knowledge and helps you with specialized expertise!

---

## Skill Categories

### 🤖 AI & Agents (`/ai_agents`)
Everything related to Artificial Intelligence, Large Language Models, and Autonomous Agents.
-   **Agents:** `@agent-architect`, `@crewai`, `@subagent-driven-development`
-   **LLMs:** `@prompt-engineering`, `@rag-implementation`, `@notebooklm`
-   **Bots:** `@telegram-bot-builder`, `@discord-bot-architect`

### 🛡️ Security & Pentesting (`/security`)
Tools and methodologies for security audits, penetration testing, and vulnerability assessment.
-   **Web Sec:** `@burp-suite-testing`, `@xss-html-injection`, `@sql-injection-testing`
-   **Infra Sec:** `@aws-penetration-testing`, `@linux-privilege-escalation`
-   **Red Team:** `@red-team-tactics`, `@ethical-hacking-methodology`

### 💻 Development (`/development`)
Core software development skills covering languages, frontend/backend frameworks, and databases.
-   **Frontend:** `@react-best-practices`, `@nextjs-best-practices`, `@tailwind-patterns`
-   **Backend:** `@python-patterns`, `@nestjs-expert`, `@postgres-best-practices`
-   **Integrations:** `@stripe-integration`, `@firebase`, `@plaid-fintech`

### ☁️ DevOps & Infrastructure (`/devops`)
Infrastructure as Code, Cloud Providers, System Administration, and CI/CD.
-   **Cloud:** `@aws-serverless`, `@gcp-cloud-run`, `@azure-functions`
-   **Tools:** `@docker-expert`, `@git-pushing`, `@linux-shell-scripting`

### 🎨 Product & Design (`/product_design`)
Product management, visual design, user experience, and growth marketing.
-   **Design:** `@ui-ux-pro-max`, `@canvas-design`, `@mobile-design`
-   **Growth:** `@seo-audit`, `@copywriting`, `@launch-strategy`, `@ab-test-setup`

### 🏗️ Engineering Practices (`/engineering`)
Best practices, processes, architecture, and software quality assurance.
-   **Process:** `@clean-code`, `@code-review-checklist`, `@writing-plans`
-   **Testing:** `@test-driven-development`, `@systematic-debugging`
-   **Planning:** `@concise-planning`, `@architecture`

### 📂 Office & Productivity (`/office`)
Automation and creation of standard office documents.
-   `@docx` / `@xlsx` / `@pptx` - Deep manipulation of Office files
-   `@pdf` - PDF processing and form handling

---

## Finding Skills

### Method 1: Browse by Category
Navigate into the folder that matches your intent (e.g., `cd skills/security`).

### Method 2: Search by keyword
```bash
# Recursive search for a skill name
find . -name "*skill-name*"
```

---

## Creating Your Own Skill

Want to create a new skill? Check out:
1. [CONTRIBUTING.md](../CONTRIBUTING.md) - How to contribute
2. [docs/SKILL_ANATOMY.md](../docs/SKILL_ANATOMY.md) - Skill structure guide
3. `@skill-creator` - Use this skill to create new skills!

---

## Documentation

- **[Getting Started](../GETTING_STARTED.md)** - Quick start guide
- **[Examples](../docs/EXAMPLES.md)** - Real-world usage examples
- **[FAQ](../FAQ.md)** - Common questions

---

## References

- [Anthropic Skills](https://github.com/anthropic/skills)
- [UI/UX Pro Max Skills](https://github.com/nextlevelbuilder/ui-ux-pro-max-skill)
