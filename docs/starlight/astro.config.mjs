import { defineConfig } from 'astro/config';

import starlight from "@astrojs/starlight";

// https://astro.build/config
export default defineConfig({
  integrations: [starlight(
    {
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
        defaultLocale: 'en',
        locales: {
            en: {
                label: 'English',
            }
        }
    }
  )]
});