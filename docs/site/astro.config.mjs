// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// https://astro.build/config
export default defineConfig({
	site: 'https://lliquid.github.io',
	base: '/agentcore-rl-toolkit',
	integrations: [
		starlight({
			title: 'ART',
			description: 'RL training on top of Bedrock AgentCore Runtime.',
			customCss: ['./src/styles/custom.css'],
			head: [
				{
					tag: 'script',
					attrs: { is: 'inline' },
					content: `
						try {
							if (!localStorage.getItem('starlight-theme')) {
								localStorage.setItem('starlight-theme', 'dark');
							}
							document.documentElement.dataset.theme =
								localStorage.getItem('starlight-theme') || 'dark';
						} catch {
							document.documentElement.dataset.theme = 'dark';
						}
					`,
				},
			],
			social: [
				{
					icon: 'github',
					label: 'GitHub',
					href: 'https://github.com/awslabs/agentcore-rl-toolkit',
				},
			],
			sidebar: [
				{ label: 'Overview', slug: 'guides/overview' },
				{
					label: 'Setup Guide',
					items: [
						{ label: 'Prepare agent for RL', slug: 'guides/agent-adaptation' },
						{
							label: 'Training backends',
							items: [
								{ label: 'slime', slug: 'guides/slime-backend-setup' },
								{ label: 'rllm', slug: 'guides/rllm-backend-setup' },
								{ label: 'verl', slug: 'guides/verl-backend-setup' },
							],
						},
					],
				},
				{
					label: 'Examples',
					items: [
						{ label: 'Overview', slug: 'examples' },
						{
							label: 'Strands Agents on AgentCore',
							items: [
								{ label: 'Math', slug: 'examples/strands-math-agent' },
								{ label: 'AppWorld', slug: 'examples/strands-appworld-agent' },
								{ label: 'Migration', slug: 'examples/strands-migration-agent' },
								{ label: 'OfficeBench', slug: 'examples/strands-officebench-agent' },
							],
						},
					],
				},
				{
					label: 'API Reference',
					items: [
						{
							label: 'Core',
							autogenerate: { directory: 'api/core' },
						},
						{
							label: 'Backends',
							items: [
								{
									label: 'slime',
									autogenerate: { directory: 'api/backends/slime' },
								},
							],
						},
					],
				},
				{
					label: 'Troubleshooting',
					items: [
						{
							label: 'Training backends',
							items: [
								{ label: 'slime', slug: 'troubleshooting/slime' },
								{ label: 'rllm', slug: 'troubleshooting/rllm' },
								{ label: 'verl', slug: 'troubleshooting/verl' },
							],
						},
					],
				},
			],
		}),
	],
});
