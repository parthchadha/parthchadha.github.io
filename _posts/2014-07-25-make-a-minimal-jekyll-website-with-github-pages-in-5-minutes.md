---
layout: post
title: How to set up a minimal Jekyll website with GitHub pages in 5 minutes
summary: A tutorial on how to make a minimal, responsive Jekyll website, just like this.
tags: how-to github-pages jekyll
---

After spending a whole day (a few other tasks here and there) to set up this website, I think I owe it to the internet - a tutorial on how one can set up a website, just like this, in 5 minutes! So read on if you liked the way I've put together this website.

P.S.: This is my first blog post, ever...

## Pre-requisites

- A Unix based machine
- A [GitHub](https://github.com/) account; and subsequently, knowledge on how to use git.
- [Ruby](https://www.ruby-lang.org/en/downloads/)
- [RubyGems](http://rubygems.org/pages/download)

## Deploy with GitHub Pages

The good guys at GitHub let you host your own static website on their servers through [GitHub Pages](https://pages.github.com/). Just create a repository in your GitHub account with the name ***username*.github.io** (where ***username*** is your GitHub username) and you're good to proceed.

## Get Lanyon theme for Jekyll

"[Lanyon](http://lanyon.getpoole.com/) is an unassuming Jekyll theme that places content first by tucking away navigation in a hidden drawer. It's based on Poole, the Jekyll butler."

Enter these commands on the terminal in a directory of your choice after replacing *username* with your GitHub username:

<script src="https://gist.github.com/nilakshdas/01efc6867688c5fc144d.js"></script>

GitHub will automatically deploy the sample Lanyon template to ***username*.github.io**! Your website is now live.

## Install and run Jekyll locally

It really just takes one line to install Jekyll and all it's dependencies:

<script src="https://gist.github.com/nilakshdas/8665cb1a3adc1fb2b117.js"></script>

... and another line to deploy it locally (when you're in the website's root folder):

<script src="https://gist.github.com/nilakshdas/c5ea9b109198d4940cbc.js"></script>

After you run this, you can access your website locally at [http://127.0.0.1:4000](http://127.0.0.1:4000)

<br/>

Now you're all set to tinker with your new website. Make sure you have a look at the [Jekyll docs](http://jekyllrb.com/docs/home/) to unleash the full power of Jekyll.