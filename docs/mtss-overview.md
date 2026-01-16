# Maritime Technical Support System (MTSS)

## Overview

The Maritime Technical Support System (MTSS) is an AI-powered knowledge management and support platform designed specifically for the maritime industry. It helps vessel crew, technical officers, and shore-based maintenance teams find solutions to technical issues by searching through historical records and documentation.

## Purpose

MTSS addresses a common challenge in maritime operations: accessing relevant historical information quickly when troubleshooting vessel issues. Instead of manually searching through years of emails, maintenance reports, and technical documents, crew members can ask natural language questions and receive answers with source citations.

## Key Features

### Issue History Search
- Search for similar past issues and how they were resolved
- Find patterns in recurring equipment problems across the fleet
- Access maintenance reports and incident documentation

### Technical Knowledge Base
- Search technical documentation and procedures
- Find manufacturer guidelines and specifications
- Access operational procedures and best practices

### AI-Powered Assistance
- Natural language queries - ask questions as you would to a colleague
- Source attribution - every answer cites the original documents
- Pattern recognition - identify systemic issues across multiple vessels

## Target Users

- **Vessel Crew** - Chief Engineers, Officers, and technical staff seeking troubleshooting guidance
- **Technical Superintendents** - Shore-based technical managers researching fleet-wide issues
- **Maintenance Teams** - Personnel planning repairs and preventive maintenance
- **Operations Staff** - Anyone needing quick access to historical operational data

## Data Sources

MTSS indexes and searches through:
- Email archives (correspondence between vessels and shore offices)
- Maintenance reports and work orders
- Technical documentation and manuals
- Incident reports and root cause analyses
- Operational logs and vessel communications

## How It Works

1. **Data Ingestion** - Documents are processed and indexed using advanced embedding techniques
2. **Semantic Search** - User queries are matched against the knowledge base using AI-powered similarity search
3. **Retrieval Augmented Generation (RAG)** - Relevant documents are retrieved and used to generate accurate, cited responses
4. **Source Attribution** - Every answer includes references to the original documents for verification

## Example Use Cases

| Scenario | How MTSS Helps |
|----------|----------------|
| Steering gear hydraulic leak | Find past cases with similar symptoms and their resolutions |
| Generator startup failure | Search for troubleshooting procedures and past incident reports |
| Turbocharger maintenance | Locate maintenance procedures and manufacturer recommendations |
| Fleet-wide pattern analysis | Identify if similar issues occurred on other vessels |

## Technical Architecture

- **Frontend**: Next.js web application with responsive design
- **Backend**: Python API with Pydantic AI agent
- **Search**: Vector database with semantic search capabilities
- **AI Model**: Large Language Model for response generation
- **Authentication**: Supabase-based user authentication

## Getting Started

1. Sign in with your authorized credentials
2. Type your question in natural language
3. Review the AI-generated response with source citations
4. Click on sources to access original documents

---

*MTSS v0.1.0 | Developed for MTSS*
