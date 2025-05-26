---
title: 'Migrating 600 Million Data using Airflow and PySpark'
date: 2025-01-05T01:52:14+07:00
tags: [data, DE, ETL]
draft: true
description: ""
disableShare: true
hideSummary: false
ShowReadingTime: true
ShowWordCount: true
cover:
    image: "<image path/url>" # image path/url
    alt: "<alt text>" # alt text
    caption: "<text>" # display caption under cover
    relative: true # when using page bundles set this to true
math: katex
keywords: [ETL, data engineering]
summary: "I was asked to migrate 600M+ data from our internal MySQL DB to BigQuery. Here is how I did it using incremental load strategy."
---

### Background
We have an MySQL table that holds 600 million data since 2023. 