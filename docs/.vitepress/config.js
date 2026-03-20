import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'MiniTorchBR',
  description: 'A lightweight autograd framework for AI training',
  base: '/MiniTorch/',

  head: [
    ['link', { rel: 'icon', href: '/MiniTorch/favicon.ico' }]
  ],

  themeConfig: {
    logo: null,
    siteTitle: 'MiniTorchBR',

    nav: [
      { text: 'Guide', link: '/guide/getting-started' },
      { text: 'API', link: '/api/core' },
      { text: 'Examples', link: '/examples/basic-autograd' },
      {
        text: 'v0.3.2',
        items: [
          { text: 'Changelog', link: 'https://github.com/BriceLucifer/MiniTorch/releases' },
          { text: 'PyPI', link: 'https://pypi.org/project/minitorchbr/' }
        ]
      }
    ],

    sidebar: {
      '/guide/': [
        {
          text: 'Introduction',
          items: [
            { text: 'Getting Started', link: '/guide/getting-started' },
            { text: 'Autograd System', link: '/guide/autograd' },
            { text: 'Neural Networks', link: '/guide/neural-networks' }
          ]
        }
      ],
      '/api/': [
        {
          text: 'API Reference',
          items: [
            { text: 'Core (Variable)', link: '/api/core' },
            { text: 'Operations', link: '/api/ops' },
            { text: 'nn (Layers)', link: '/api/nn' },
            { text: 'optim (Optimizers)', link: '/api/optim' },
            { text: 'data (Loaders)', link: '/api/data' }
          ]
        }
      ],
      '/examples/': [
        {
          text: 'Examples',
          items: [
            { text: 'Basic Autograd', link: '/examples/basic-autograd' },
            { text: 'Training a Network', link: '/examples/training' },
            { text: 'MNIST Classifier', link: '/examples/mnist' },
            { text: 'Graph Visualization', link: '/examples/visualization' }
          ]
        }
      ]
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/BriceLucifer/MiniTorch' }
    ],

    footer: {
      message: 'Released under the MIT License.',
      copyright: 'Copyright © 2024 BriceLucifer'
    },

    search: {
      provider: 'local'
    }
  }
})
