import { defineConfig } from 'astro/config';

import starlight from "@astrojs/starlight";

// https://astro.build/config
export default defineConfig({
  site: 'https://gagnonanthony.github.io',
  base: '/NeuroStatX',
  integrations: [starlight(
    {
        favicon: "/public/favicon-32x32.png",
        head: [
            {
                tag: 'link',
                attrs: {
                    rel: 'icon',
                    href: '/public/favicon-32x32.png',
                    sizes: '32x32',
                }
            }
        ],
        sidebar: [
            {
                label: "Getting Started",
                autogenerate: {directory: "getting-started"},
            },
            {
                label: "Tutorials",
                items: [
                    { label: "Introduction to NeuroStatX", link: "./tutorials/intro" },
                    { label: "Applying Fuzzy Clustering", link: "./tutorials/fuzzyclustering" },
                ]
            },
            {
                label: "Documentation",
                autogenerate: {directory: "API"},
            }
        ],
        title: "NeuroStatX Documentation",
        logo: {
            light: "./src/assets/logo_with_text.svg",
            dark: "./src/assets/white_logo_with_text.svg",
            replacesTitle: true,
        },
            customCss: [
            './src/styles/custom.css',
            './src/fonts/font-face.css'
        ],
        social: {
            github: 'https://github.com/gagnonanthony/NeuroStatX'
        },
        defaultLocale: '',
        }
  )]
});