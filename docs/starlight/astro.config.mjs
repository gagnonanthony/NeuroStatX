import { defineConfig } from 'astro/config';

import starlight from "@astrojs/starlight";
import starlightPydocs, { pydocsSidebarGroup } from "starlight-pydocs";

// https://astro.build/config
export default defineConfig({
  site: 'https://gagnonanthony.github.io',
  base: '/NeuroStatX/',
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
        plugins: [
            starlightPydocs({
                packages: [{
                    name: 'neurostatx',
                    base: 'api/neurostatx',
                    search: ['../..'],
                    docstringStyle: 'numpy',
                    docstringOptions: { warn_unknown_params: false },
                    filters: {
                        inherited: false,
                        private: false,
                        imported: false,
                        special: false,
                    },
                    members: {
                        exclude: [
                            'neurostatx.clustering.fuzzy.process_cluster',
                            'neurostatx.clustering.viz.create_radar_plot',
                            'neurostatx.statistics.models.permutation_test',
                            'neurostatx.utils.factor.FormattedTextPrompt',
                            'neurostatx.statistics.harmonization.make_design_matrix',
                            'neurostatx.statistics.harmonization.standardize_across_features',
                            'neurostatx.statistics.harmonization.aprior',
                            'neurostatx.statistics.harmonization.bprior',
                            'neurostatx.statistics.harmonization.postmean',
                            'neurostatx.statistics.harmonization.postvar',
                            'neurostatx.statistics.harmonization.convert_zeroes',
                            'neurostatx.statistics.harmonization.fit_LS_model_and_find_priors',
                            'neurostatx.statistics.harmonization.it_sol',
                            'neurostatx.statistics.harmonization.int_eprior',
                            'neurostatx.statistics.harmonization.find_parametric_adjustments',
                            'neurostatx.statistics.harmonization.find_non_parametric_adjustments',
                            'neurostatx.statistics.harmonization.find_non_eb_adjustments',
                            'neurostatx.statistics.harmonization.adjust_data_final',
                            'neurostatx.clustering.distance',
                            'neurostatx.clustering.distance.**',
                            'neurostatx.io.download',
                            'neurostatx.io.download.**',
                            'neurostatx.io.utils',
                            'neurostatx.io.utils.**',
                        ],
                    },
                    sourceLink: {
                        host: 'github',
                        repo: 'gagnonanthony/NeuroStatX',
                        ref: 'main',
                        root: '../..',
                    },
                }],
                inventories: [
                    'python',
                    { url: 'https://scikit-learn.org/stable/objects.inv' },
                ],
            }),
        ],
        sidebar: [
            {
                label: "Getting Started",
                items: [
                    { autogenerate: {directory: "getting-started"} }
                ]
            },
            {
                label: "Tutorials",
                items: [
                    { label: "Introduction to NeuroStatX", link: "tutorials/intro" },
                    { label: "Applying Fuzzy Clustering", link: "tutorials/fuzzyclustering" },
                ]
            },
            {
                label: "API Reference",
                items: [pydocsSidebarGroup]
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
        social: [
            { icon: 'github', label: 'GitHub', href: 'https://github.com/gagnonanthony/NeuroStatX' }
        ],
        defaultLocale: '',
        }
  )]
});
