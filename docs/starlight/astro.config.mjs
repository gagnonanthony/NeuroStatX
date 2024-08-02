import { defineConfig } from 'astro/config';

import starlight from "@astrojs/starlight";

// https://astro.build/config
export default defineConfig({
  site: 'https://gagnonanthony.github.io',
  base: '/NeuroStatX/',
  integrations: [starlight(
    {
        sidebar: [
            {
                label: "Getting Started",
                autogenerate: {directory: "Getting Started"},
            },
            {
                label: "Tutorials",
                autogenerate: {directory: "Tutorials"},
            },
            {
                label: "API Reference",
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